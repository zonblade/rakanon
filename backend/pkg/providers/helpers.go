package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"pentagi/pkg/cast"
	"pentagi/pkg/csum"
	"pentagi/pkg/database"
	"pentagi/pkg/docker"
	"pentagi/pkg/providers/provider"
	"pentagi/pkg/templates"
	"pentagi/pkg/tools"

	"github.com/sirupsen/logrus"
	"github.com/vxcontrol/langchaingo/llms"
)

const (
	RepeatingToolCallThreshold   = 3
	maxQASectionsAfterRestore    = 3
	keepQASectionsAfterRestore   = 1
	lastSecBytesAfterRestore     = 16 * 1024 // 16 KB
	maxBPBytesAfterRestore       = 8 * 1024  // 8 KB
	maxQABytesAfterRestore       = 20 * 1024 // 20 KB
	msgLogResultSummarySizeLimit = 50 * 1024 // 50 KB
	msgLogResultEntrySizeLimit   = 1024      // 1 KB
)

type repeatingDetector struct {
	funcCalls []llms.FunctionCall
}

func (rd *repeatingDetector) detect(toolCall llms.ToolCall) bool {
	if toolCall.FunctionCall == nil {
		return false
	}

	funcCall := rd.clearCallArguments(toolCall.FunctionCall)

	if len(rd.funcCalls) == 0 {
		rd.funcCalls = append(rd.funcCalls, funcCall)
		return false
	}

	lastToolCall := rd.funcCalls[len(rd.funcCalls)-1]
	if lastToolCall.Name != funcCall.Name || lastToolCall.Arguments != funcCall.Arguments {
		rd.funcCalls = []llms.FunctionCall{funcCall}
		return false
	}

	rd.funcCalls = append(rd.funcCalls, funcCall)

	return len(rd.funcCalls) >= RepeatingToolCallThreshold
}

func (rd *repeatingDetector) clearCallArguments(toolCall *llms.FunctionCall) llms.FunctionCall {
	var v map[string]any
	if err := json.Unmarshal([]byte(toolCall.Arguments), &v); err != nil {
		return *toolCall
	}

	delete(v, "message")
	var keys []string
	for k := range v {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var buffer strings.Builder
	for _, k := range keys {
		buffer.WriteString(fmt.Sprintf("%s: %v\n", k, v[k]))
	}

	return llms.FunctionCall{
		Name:      toolCall.Name,
		Arguments: buffer.String(),
	}
}

func (fp *flowProvider) getTasksInfo(ctx context.Context, taskID int64) (*tasksInfo, error) {
	var (
		err  error
		info tasksInfo
	)

	info.Tasks, err = fp.db.GetFlowTasks(ctx, fp.flowID)
	if err != nil {
		return nil, fmt.Errorf("failed to get flow %d tasks: %w", fp.flowID, err)
	}

	for idx, t := range info.Tasks {
		if t.ID == taskID {
			info.Task = t
			info.Tasks = append(info.Tasks[:idx], info.Tasks[idx+1:]...)
			break
		}
	}

	info.Subtasks, err = fp.db.GetFlowSubtasks(ctx, fp.flowID)
	if err != nil {
		return nil, fmt.Errorf("failed to get flow %d subtasks: %w", fp.flowID, err)
	}

	return &info, nil
}

func (fp *flowProvider) getSubtasksInfo(taskID int64, subtasks []database.Subtask) *subtasksInfo {
	var info subtasksInfo
	for _, subtask := range subtasks {
		if subtask.TaskID != taskID && taskID != 0 {
			continue
		}

		switch subtask.Status {
		case database.SubtaskStatusCreated:
			info.Planned = append(info.Planned, subtask)
		case database.SubtaskStatusFinished, database.SubtaskStatusFailed:
			info.Completed = append(info.Completed, subtask)
		default:
			info.Subtask = &subtask
		}
	}

	return &info
}

func (fp *flowProvider) updateMsgChainResult(chain []llms.MessageContent, name, result string) ([]llms.MessageContent, error) {
	if len(chain) == 0 {
		return []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, result)}, nil
	}

	ast, err := cast.NewChainAST(chain, true)
	if err != nil {
		return nil, fmt.Errorf("failed to create chain ast: %w", err)
	}

	lastSection := ast.Sections[len(ast.Sections)-1]
	if len(lastSection.Body) == 0 {
		ast.AppendHumanMessage(result)
		return ast.Messages(), nil
	}

	lastBody := lastSection.Body[len(lastSection.Body)-1]
	switch lastBody.Type {
	case cast.Completion, cast.Summarization:
		ast.AppendHumanMessage(result)
		return ast.Messages(), nil
	case cast.RequestResponse:
		for _, msg := range lastBody.ToolMessages {
			for pdx, part := range msg.Parts {
				toolCallResp, ok := part.(llms.ToolCallResponse)
				if !ok {
					continue
				}

				if toolCallResp.Name == name {
					toolCallResp.Content = result
					msg.Parts[pdx] = toolCallResp
					return ast.Messages(), nil
				}
			}
		}

		ast.AppendHumanMessage(result)
		return ast.Messages(), nil
	default:
		return nil, fmt.Errorf("unknown message type: %d", lastBody.Type)
	}
}

// Makes chain consistent by adding default responses for any pending tool calls
func (fp *flowProvider) ensureChainConsistency(chain []llms.MessageContent) ([]llms.MessageContent, error) {
	if len(chain) == 0 {
		return chain, nil
	}

	ast, err := cast.NewChainAST(chain, true)
	if err != nil {
		return nil, fmt.Errorf("failed to create chain ast: %w", err)
	}

	return ast.Messages(), nil
}

func (fp *flowProvider) getTaskPrimaryAgentChainSummary(
	ctx context.Context,
	taskID int64,
	summarizerHandler tools.SummarizeHandler,
) (string, error) {
	msgChain, err := fp.db.GetFlowTaskTypeLastMsgChain(ctx, database.GetFlowTaskTypeLastMsgChainParams{
		FlowID: fp.flowID,
		TaskID: database.Int64ToNullInt64(&taskID),
		Type:   database.MsgchainTypePrimaryAgent,
	})
	if err != nil || isEmptyChain(msgChain.Chain) {
		return "", fmt.Errorf("failed to get task primary agent chain: %w", err)
	}

	chain := []llms.MessageContent{}
	if err := json.Unmarshal(msgChain.Chain, &chain); err != nil {
		return "", fmt.Errorf("failed to unmarshal task primary agent chain: %w", err)
	}

	ast, err := cast.NewChainAST(chain, true)
	if err != nil {
		return "", fmt.Errorf("failed to create refiner chain ast: %w", err)
	}

	var humanMessages, aiMessages []llms.MessageContent
	for _, section := range ast.Sections {
		if section.Header.HumanMessage != nil {
			humanMessages = append(humanMessages, *section.Header.HumanMessage)
		}
		for _, pair := range section.Body {
			aiMessages = append(aiMessages, pair.Messages()...)
		}
	}

	humanSummary, err := csum.GenerateSummary(ctx, summarizerHandler, humanMessages, nil)
	if err != nil {
		return "", fmt.Errorf("failed to generate human summary: %w", err)
	}

	aiSummary, err := csum.GenerateSummary(ctx, summarizerHandler, humanMessages, aiMessages)
	if err != nil {
		return "", fmt.Errorf("failed to generate ai summary: %w", err)
	}

	summary := fmt.Sprintf(`## Task Summary

### User Requirements
*Summarized input from user:*

%s

### Execution Results
*Summarized actions and outcomes:*

%s`, humanSummary, aiSummary)
	return summary, nil
}


func (fp *flowProvider) getTaskMsgLogsSummary(
	ctx context.Context,
	taskID int64,
	summarizerHandler tools.SummarizeHandler,
) (string, error) {
	msgLogs, err := fp.db.GetTaskMsgLogs(ctx, database.Int64ToNullInt64(&taskID))
	if err != nil {
		return "", fmt.Errorf("failed to get task msg logs: %w", err)
	}

	if len(msgLogs) == 0 {
		// LOG NO MSG LOGS
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-msg-logs-summary",
			"action":    "no_msg_logs",
			"flow":      1,
			"task_id":   taskID,
			"result":    "no msg logs",
		}).Info("=== CONTEXT MANAGEMENT: No Message Logs Available ===")
		
		return "no msg logs", nil
	}

	// LOG MSG LOGS RETRIEVED
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-msg-logs-summary",
		"action":    "msg_logs_retrieved",
		"flow":      1,
		"task_id":   taskID,
		"logs_count": len(msgLogs),
		"raw_logs":  msgLogs,
	}).Info("=== CONTEXT MANAGEMENT: Message Logs Retrieved ===")

	// truncate msg logs result to cut down the size the message to summarize
	originalLogsLen := len(msgLogs)
	for _, msgLog := range msgLogs {
		if len(msgLog.Result) > msgLogResultEntrySizeLimit {
			msgLog.Result = msgLog.Result[:msgLogResultEntrySizeLimit] + textTruncateMessage
		}
	}

	// === EXECUTION LOGS PROMPT ===
	executionLogsParams := map[string]any{
		"MsgLogs": msgLogs,
	}

	message, err := fp.prompter.RenderTemplate(templates.PromptTypeExecutionLogs, executionLogsParams)
	if err != nil {
		return "", fmt.Errorf("failed to render task msg logs template: %w", err)
	}

	// LOG EXECUTION LOGS PROMPT
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-msg-logs-summary",
		"action":    "execution_logs_prompt",
		"flow":      1,
		"prompt":    message,
		"params":    executionLogsParams,
		"prompt_type": templates.PromptTypeExecutionLogs,
		"task_id":   taskID,
		"logs_count": len(msgLogs),
		"message_length": len(message),
	}).Info("=== PROMPT GENERATION: Execution Logs Template Rendered ===")

	// Handle template size limiting
	for l := len(msgLogs) / 2; l > 2; l /= 2 {
		if len(message) < msgLogResultSummarySizeLimit {
			break
		}

		msgLogs = msgLogs[l:]
		executionLogsParams["MsgLogs"] = msgLogs
		message, err = fp.prompter.RenderTemplate(templates.PromptTypeExecutionLogs, executionLogsParams)
		if err != nil {
			return "", fmt.Errorf("failed to render task msg logs template: %w", err)
		}

		// LOG EXECUTION LOGS PROMPT TRUNCATED
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-msg-logs-summary",
			"action":    "execution_logs_prompt_truncated",
			"flow":      1,
			"prompt":    message,
			"params":    executionLogsParams,
			"prompt_type": templates.PromptTypeExecutionLogs,
			"task_id":   taskID,
			"original_logs_count": originalLogsLen,
			"truncated_logs_count": len(msgLogs),
			"truncation_level": l,
			"message_length": len(message),
			"size_limit": msgLogResultSummarySizeLimit,
		}).Info("=== PROMPT GENERATION: Execution Logs Template Truncated ===")
	}

	summary, err := summarizerHandler(ctx, message)
	if err != nil {
		return "", fmt.Errorf("failed to summarize task msg logs: %w", err)
	}

	// LOG MSG LOGS SUMMARY COMPLETE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-msg-logs-summary",
		"action":    "msg_logs_summary_complete",
		"flow":      1,
		"summary":   summary,
		"original_message": message,
		"original_length": len(message),
		"summary_length": len(summary),
		"compression_ratio": float64(len(summary)) / float64(len(message)),
		"task_id":   taskID,
		"logs_processed": len(msgLogs),
	}).Info("=== CONTEXT MANAGEMENT: Message Logs Summary Complete ===")

	return summary, nil
}


func (fp *flowProvider) restoreChain(
	ctx context.Context,
	taskID, subtaskID *int64,
	optAgentType provider.ProviderOptionsType,
	msgChainType database.MsgchainType,
	systemPrompt, humanPrompt string,
) (int64, []llms.MessageContent, error) {
	var chain []llms.MessageContent
	fallback := func() {
		chain = []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeSystem, systemPrompt),
		}
		if humanPrompt != "" {
			chain = append(chain, llms.TextParts(llms.ChatMessageTypeHuman, humanPrompt))
		}

		// LOG CHAIN FALLBACK CREATED
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-chain-restoration",
			"action":    "chain_fallback_created",
			"flow":      1,
			"fallback_chain": chain,
			"system_prompt": systemPrompt,
			"human_prompt": humanPrompt,
			"agent_type": optAgentType,
			"chain_type": msgChainType,
			"task_id":   taskID,
			"subtask_id": subtaskID,
		}).Info("=== CHAIN MANAGEMENT: Fallback Chain Created ===")
	}

	msgChain, err := fp.db.GetFlowTaskTypeLastMsgChain(ctx, database.GetFlowTaskTypeLastMsgChainParams{
		FlowID: fp.flowID,
		TaskID: database.Int64ToNullInt64(taskID),
		Type:   msgChainType,
	})
	if err != nil || isEmptyChain(msgChain.Chain) {
		fallback()
	} else {
		err = json.Unmarshal(msgChain.Chain, &chain)
		if err != nil {
			// LOG CHAIN UNMARSHAL FAILURE
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-chain-restoration",
				"action":    "chain_unmarshal_failure",
				"flow":      1,
				"error":     err.Error(),
				"msg_chain_id": msgChain.ID,
				"agent_type": optAgentType,
				"chain_type": msgChainType,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Warn("=== CHAIN MANAGEMENT: Chain Unmarshal Failed - Using Fallback ===")
			
			fallback()
		} else {
			// LOG EXISTING CHAIN RESTORED
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-chain-restoration",
				"action":    "existing_chain_restored",
				"flow":      1,
				"restored_chain": chain,
				"restored_chain_length": len(chain),
				"msg_chain_id": msgChain.ID,
				"agent_type": optAgentType,
				"chain_type": msgChainType,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== CHAIN MANAGEMENT: Existing Chain Restored ===")
		}
	}

	chainBlob, err := json.Marshal(chain)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to marshal msg chain: %w", err)
	}

	chainRecord, err := fp.db.CreateMsgChain(ctx, database.CreateMsgChainParams{
		Type:          msgChainType,
		Model:         fp.Model(optAgentType),
		ModelProvider: string(fp.Type()),
		Chain:         chainBlob,
		FlowID:        fp.flowID,
		TaskID:        database.Int64ToNullInt64(taskID),
		SubtaskID:     database.Int64ToNullInt64(subtaskID),
	})
	if err != nil {
		return 0, nil, fmt.Errorf("failed to create msg chain: %w", err)
	}

	// LOG CHAIN RECORD CREATED
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-chain-restoration",
		"action":    "chain_record_created",
		"flow":      1,
		"final_chain": chain,
		"final_chain_length": len(chain),
		"new_msg_chain_id": chainRecord.ID,
		"agent_type": optAgentType,
		"chain_type": msgChainType,
		"model": fp.Model(optAgentType),
		"task_id":   taskID,
		"subtask_id": subtaskID,
	}).Info("=== CHAIN MANAGEMENT: Chain Record Created ===")

	return chainRecord.ID, chain, nil
}

// Eliminates code duplication by abstracting database operations on message chains
func (fp *flowProvider) processChain(
	ctx context.Context,
	msgChainID int64,
	logger *logrus.Entry,
	transform func([]llms.MessageContent) ([]llms.MessageContent, error),
) error {
	msgChain, err := fp.db.GetMsgChain(ctx, msgChainID)
	if err != nil {
		logger.WithError(err).Error("failed to get message chain")
		return fmt.Errorf("failed to get message chain %d: %w", msgChainID, err)
	}

	var chain []llms.MessageContent
	if err := json.Unmarshal(msgChain.Chain, &chain); err != nil {
		logger.WithError(err).Error("failed to unmarshal message chain")
		return fmt.Errorf("failed to unmarshal message chain %d: %w", msgChainID, err)
	}

	updatedChain, err := transform(chain)
	if err != nil {
		logger.WithError(err).Error("failed to transform chain")
		return fmt.Errorf("failed to transform chain: %w", err)
	}

	chainBlob, err := json.Marshal(updatedChain)
	if err != nil {
		logger.WithError(err).Error("failed to marshal updated chain")
		return fmt.Errorf("failed to marshal updated chain %d: %w", msgChainID, err)
	}

	_, err = fp.db.UpdateMsgChain(ctx, database.UpdateMsgChainParams{
		Chain: chainBlob,
		ID:    msgChainID,
	})
	if err != nil {
		logger.WithError(err).Error("failed to update message chain")
		return fmt.Errorf("failed to update message chain %d: %w", msgChainID, err)
	}

	return nil
}

func (fp *flowProvider) prepareExecutionContext(ctx context.Context, taskID, subtaskID int64) (string, error) {
	tasksInfo, err := fp.getTasksInfo(ctx, taskID)
	if err != nil {
		return "", fmt.Errorf("failed to get tasks info: %w", err)
	}

	subtasksInfo := fp.getSubtasksInfo(taskID, tasksInfo.Subtasks)
	if subtasksInfo.Subtask == nil {
		// CREATE THE SUBTASKS SLICE HERE
		subtasks := make([]database.Subtask, 0, len(subtasksInfo.Planned)+len(subtasksInfo.Completed))
		subtasks = append(subtasks, subtasksInfo.Planned...)
		subtasks = append(subtasks, subtasksInfo.Completed...)
		slices.SortFunc(subtasks, func(a, b database.Subtask) int {
			return int(a.ID - b.ID)
		})

		for i, subtask := range subtasks {
			if subtask.ID == subtaskID {
				subtasksInfo.Subtask = &subtask
				subtasksInfo.Planned = subtasks[i+1:]
				subtasksInfo.Completed = subtasks[:i]  // <-- YOUR LINE HERE
				break
			}
		}
	}

	// === FULL EXECUTION CONTEXT PROMPT ===
	executionContextParams := map[string]any{
		"Task":              tasksInfo.Task,
		"Tasks":             tasksInfo.Tasks,
		"CompletedSubtasks": subtasksInfo.Completed,
		"Subtask":           subtasksInfo.Subtask,
		"PlannedSubtasks":   subtasksInfo.Planned,
	}

	executionContextRaw, err := fp.prompter.RenderTemplate(templates.PromptTypeFullExecutionContext, executionContextParams)
	if err != nil {
		return "", fmt.Errorf("failed to render execution context: %w", err)
	}

	// LOG FULL EXECUTION CONTEXT PROMPT
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-execution-context-preparation",
		"action":    "full_execution_context_prompt",
		"flow":      1,
		"prompt":    executionContextRaw,
		"params":    executionContextParams,
		"prompt_type": templates.PromptTypeFullExecutionContext,
		"task_id":   taskID,
		"subtask_id": subtaskID,
		"tasks_count": len(tasksInfo.Tasks),
		"completed_subtasks_count": len(subtasksInfo.Completed),
		"planned_subtasks_count": len(subtasksInfo.Planned),
	}).Info("=== PROMPT GENERATION: Full Execution Context Template Rendered ===")

	summarizeHandler := fp.GetSummarizeResultHandler(&taskID, &subtaskID)
	executionContext, err := summarizeHandler(ctx, executionContextRaw)
	if err != nil {
		return "", fmt.Errorf("failed to summarize execution context: %w", err)
	}

	// LOG EXECUTION CONTEXT SUMMARIZED
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-execution-context-preparation",
		"action":    "execution_context_summarized",
		"flow":      1,
		"original_context": executionContextRaw,
		"summarized_context": executionContext,
		"original_length": len(executionContextRaw),
		"summarized_length": len(executionContext),
		"compression_ratio": float64(len(executionContext)) / float64(len(executionContextRaw)),
		"task_id":   taskID,
		"subtask_id": subtaskID,
	}).Info("=== CONTEXT MANAGEMENT: Full Execution Context Summarized ===")

	return executionContext, nil
}


func (fp *flowProvider) getExecutionContext(ctx context.Context, taskID, subtaskID *int64) (string, error) {
	if taskID != nil && subtaskID != nil {
		return fp.getExecutionContextBySubtask(ctx, *taskID, *subtaskID)
	}

	if taskID != nil {
		return fp.getExecutionContextByTask(ctx, *taskID)
	}

	return fp.getExecutionContextByFlow(ctx)
}

func (fp *flowProvider) getExecutionContextBySubtask(ctx context.Context, taskID, subtaskID int64) (string, error) {
	subtask, err := fp.db.GetSubtask(ctx, subtaskID)
	if err == nil && subtask.TaskID == taskID && subtask.Context != "" {
		return subtask.Context, nil
	}

	return fp.getExecutionContextByTask(ctx, taskID)
}


func (fp *flowProvider) getExecutionContextByTask(ctx context.Context, taskID int64) (string, error) {
	tasksInfo, err := fp.getTasksInfo(ctx, taskID)
	if err != nil {
		return fp.getExecutionContextByFlow(ctx)
	}

	subtasksInfo := fp.getSubtasksInfo(taskID, tasksInfo.Subtasks)

	// === SHORT EXECUTION CONTEXT PROMPT BY TASK ===
	executionContextParams := map[string]any{
		"Task":              tasksInfo.Task,
		"Tasks":             tasksInfo.Tasks,
		"CompletedSubtasks": subtasksInfo.Completed,
		"Subtask":           subtasksInfo.Subtask,
		"PlannedSubtasks":   subtasksInfo.Planned,
	}

	executionContext, err := fp.prompter.RenderTemplate(templates.PromptTypeShortExecutionContext, executionContextParams)
	if err != nil {
		return fp.getExecutionContextByFlow(ctx)
	}

	// LOG SHORT EXECUTION CONTEXT BY TASK PROMPT
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-execution-context-by-task",
		"action":    "short_execution_context_by_task_prompt",
		"flow":      1,
		"prompt":    executionContext,
		"params":    executionContextParams,
		"prompt_type": templates.PromptTypeShortExecutionContext,
		"task_id":   taskID,
		"context_source": "by_task",
		"tasks_count": len(tasksInfo.Tasks),
		"completed_subtasks_count": len(subtasksInfo.Completed),
		"planned_subtasks_count": len(subtasksInfo.Planned),
	}).Info("=== PROMPT GENERATION: Short Execution Context By Task Template Rendered ===")

	return executionContext, nil
}


func (fp *flowProvider) getExecutionContextByFlow(ctx context.Context) (string, error) {
	tasks, err := fp.db.GetFlowTasks(ctx, fp.flowID)
	if err != nil {
		return "", fmt.Errorf("failed to get flow tasks: %w", err)
	}

	if len(tasks) == 0 {
		// LOG NO TASKS CONTEXT
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-execution-context-by-flow",
			"action":    "no_tasks_context",
			"flow":      1,
			"context":   "flow has no tasks, it's using in assistant mode",
			"flow_id":   fp.flowID,
		}).Info("=== CONTEXT MANAGEMENT: No Tasks - Assistant Mode Context ===")
		
		return "flow has no tasks, it's using in assistant mode", nil
	}

	subtasks, err := fp.db.GetFlowSubtasks(ctx, fp.flowID)
	if err != nil {
		return "", fmt.Errorf("failed to get flow subtasks: %w", err)
	}

	// Try from latest task backwards
	for tid := len(tasks) - 1; tid >= 0; tid-- {
		taskID := tasks[tid].ID

		subtasksInfo := fp.getSubtasksInfo(taskID, subtasks)

		// === SHORT EXECUTION CONTEXT PROMPT BY FLOW ===
		executionContextParams := map[string]any{
			"Task":              tasks[tid],
			"Tasks":             tasks,
			"CompletedSubtasks": subtasksInfo.Completed,
			"Subtask":           subtasksInfo.Subtask,
			"PlannedSubtasks":   subtasksInfo.Planned,
		}

		executionContext, err := fp.prompter.RenderTemplate(templates.PromptTypeShortExecutionContext, executionContextParams)
		if err != nil {
			// LOG TEMPLATE RENDER FAILURE
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-execution-context-by-flow",
				"action":    "template_render_failure",
				"flow":      1,
				"error":     err.Error(),
				"task_index": tid,
				"task_id":   taskID,
				"flow_id":   fp.flowID,
			}).Warn("=== CONTEXT MANAGEMENT: Short Execution Context Template Render Failed ===")
			continue
		}

		// LOG SHORT EXECUTION CONTEXT BY FLOW PROMPT
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-execution-context-by-flow",
			"action":    "short_execution_context_by_flow_prompt",
			"flow":      1,
			"prompt":    executionContext,
			"params":    executionContextParams,
			"prompt_type": templates.PromptTypeShortExecutionContext,
			"context_source": "by_flow",
			"task_index": tid,
			"task_id":   taskID,
			"flow_id":   fp.flowID,
			"tasks_count": len(tasks),
			"completed_subtasks_count": len(subtasksInfo.Completed),
			"planned_subtasks_count": len(subtasksInfo.Planned),
		}).Info("=== PROMPT GENERATION: Short Execution Context By Flow Template Rendered ===")

		return executionContext, nil
	}

	// Fallback case - no specific task
	subtasksInfo := fp.getSubtasksInfo(0, subtasks)

	// === SHORT EXECUTION CONTEXT PROMPT FALLBACK ===
	executionContextParams := map[string]any{
		"Tasks":             tasks,
		"CompletedSubtasks": subtasksInfo.Completed,
		"Subtask":           subtasksInfo.Subtask,
		"PlannedSubtasks":   subtasksInfo.Planned,
	}

	executionContext, err := fp.prompter.RenderTemplate(templates.PromptTypeShortExecutionContext, executionContextParams)
	if err != nil {
		return "", fmt.Errorf("failed to render execution context: %w", err)
	}

	// LOG SHORT EXECUTION CONTEXT FALLBACK PROMPT
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-execution-context-by-flow",
		"action":    "short_execution_context_fallback_prompt",
		"flow":      1,
		"prompt":    executionContext,
		"params":    executionContextParams,
		"prompt_type": templates.PromptTypeShortExecutionContext,
		"context_source": "fallback",
		"flow_id":   fp.flowID,
		"tasks_count": len(tasks),
		"completed_subtasks_count": len(subtasksInfo.Completed),
		"planned_subtasks_count": len(subtasksInfo.Planned),
	}).Info("=== PROMPT GENERATION: Short Execution Context Fallback Template Rendered ===")

	return executionContext, nil
}


func (fp *flowProvider) subtasksToMarkdown(subtasks []tools.SubtaskInfo) string {
	var buffer strings.Builder
	for sid, subtask := range subtasks {
		buffer.WriteString(fmt.Sprintf("# Subtask %d\n\n", sid+1))
		buffer.WriteString(fmt.Sprintf("## %s\n\n%s\n\n", subtask.Title, subtask.Description))
	}

	return buffer.String()
}

func (fp *flowProvider) getContainerPortsDescription() string {
	ports := docker.GetPrimaryContainerPorts(fp.flowID)
	var buffer strings.Builder
	buffer.WriteString("This container has the following ports which bind to the host:\n")
	for _, port := range ports {
		buffer.WriteString(fmt.Sprintf("* %s:%d -> %d/tcp (in container)\n", fp.publicIP, port, port))
	}
	if fp.publicIP == "0.0.0.0" {
		buffer.WriteString("you need to discover the public IP yourself via the following command:\n")
		buffer.WriteString("`curl -s https://api.ipify.org` or `curl -s ipinfo.io/ip` or `curl -s ifconfig.me`\n")
	}
	buffer.WriteString("you can listen these ports the container inside and receive connections from the internet.")
	return buffer.String()
}

func getCurrentTime() string {
	return time.Now().Format("2006-01-02 15:04:05")
}

func isEmptyChain(msgChain json.RawMessage) bool {
	var msgList []llms.MessageContent

	if err := json.Unmarshal(msgChain, &msgList); err != nil {
		return true
	}

	return len(msgList) == 0
}
