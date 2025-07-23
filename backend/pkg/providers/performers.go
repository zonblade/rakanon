package providers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"pentagi/pkg/database"
	"pentagi/pkg/providers/provider"
	"pentagi/pkg/tools"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/sirupsen/logrus"
)

func (fp *flowProvider) performTaskResultReporter(
	ctx context.Context,
	taskID, subtaskID *int64,
	systemReporterTmpl, userReporterTmpl, input string,
) (*tools.TaskResult, error) {
	var (
		taskResult   tools.TaskResult
		optAgentType = provider.OptionsTypeSimple
		msgChainType = database.MsgchainTypeReporter
	)

	// LOG TASK RESULT REPORTER EXECUTION START
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-task-result-reporter-execution",
		"action":    "task_result_reporter_execution_start",
		"flow":      1,
		"system_prompt": systemReporterTmpl,
		"user_prompt":   userReporterTmpl,
		"input":         input,
		"task_id":       taskID,
		"subtask_id":    subtaskID,
		"agent_type":    optAgentType,
		"chain_type":    msgChainType,
	}).Info("=== PROMPT EXECUTION: Task Result Reporter Execution Started ===")

	chain := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, systemReporterTmpl),
		llms.TextParts(llms.ChatMessageTypeHuman, userReporterTmpl),
	}

	cfg := tools.ReporterExecutorConfig{
		TaskID:    taskID,
		SubtaskID: subtaskID,
		ReportResult: func(ctx context.Context, name string, args json.RawMessage) (string, error) {
			err := json.Unmarshal(args, &taskResult)
			if err != nil {
				return "", fmt.Errorf("failed to unmarshal task result: %w", err)
			}
			
			// LOG REPORT RESULT TOOL CALL
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-task-result-reporter-execution",
				"action":    "report_result_tool_called",
				"flow":      1,
				"tool_name": name,
				"tool_args": string(args),
				"task_result": taskResult,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== TOOL EXECUTION: Report Result Tool Called ===")
			
			return "report result successfully processed", nil
		},
	}
	executor, err := fp.executor.GetReporterExecutor(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to get reporter executor: %w", err)
	}

	chainBlob, err := json.Marshal(chain)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal msg chain: %w", err)
	}

	msgChain, err := fp.db.CreateMsgChain(ctx, database.CreateMsgChainParams{
		Type:          msgChainType,
		Model:         fp.Model(optAgentType),
		ModelProvider: string(fp.Type()),
		Chain:         chainBlob,
		FlowID:        fp.flowID,
		TaskID:        database.Int64ToNullInt64(taskID),
		SubtaskID:     database.Int64ToNullInt64(subtaskID),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create msg chain: %w", err)
	}

	// LOG MSG CHAIN CREATED FOR REPORTER
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-task-result-reporter-execution",
		"action":    "msg_chain_created_reporter",
		"flow":      1,
		"msg_chain_id": msgChain.ID,
		"chain_type":   msgChainType,
		"model":        fp.Model(optAgentType),
		"task_id":      taskID,
		"subtask_id":   subtaskID,
	}).Info("=== CHAIN CREATION: Message Chain Created for Task Result Reporter ===")

	ctx = tools.PutAgentContext(ctx, msgChainType)
	err = fp.performAgentChain(ctx, optAgentType, msgChain.ID, taskID, subtaskID, chain, executor, fp.summarizer)
	if err != nil {
		return nil, fmt.Errorf("failed to get task reporter result: %w", err)
	}

	// LOG TASK RESULT REPORTER EXECUTION COMPLETE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-task-result-reporter-execution",
		"action":    "task_result_reporter_execution_complete",
		"flow":      1,
		"final_task_result": taskResult,
		"task_id":           taskID,
		"subtask_id":        subtaskID,
		"msg_chain_id":      msgChain.ID,
	}).Info("=== PROMPT EXECUTION: Task Result Reporter Execution Complete ===")

	if agentCtx, ok := tools.GetAgentContext(ctx); ok {
		fp.agentLog.PutLog(
			ctx,
			agentCtx.ParentAgentType,
			agentCtx.CurrentAgentType,
			input,
			taskResult.Result,
			taskID,
			subtaskID,
		)
	}

	return &taskResult, nil
}

func (fp *flowProvider) performSubtasksGenerator(
	ctx context.Context,
	taskID int64,
	systemGeneratorTmpl, userGeneratorTmpl, input string,
) ([]tools.SubtaskInfo, error) {
	var (
		subtaskList  tools.SubtaskList
		optAgentType = provider.OptionsTypeGenerator
		msgChainType = database.MsgchainTypeGenerator
	)

	// LOG SUBTASKS GENERATOR EXECUTION START
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-subtasks-generator-execution",
		"action":    "subtasks_generator_execution_start",
		"flow":      1,
		"system_prompt": systemGeneratorTmpl,
		"user_prompt":   userGeneratorTmpl,
		"input":         input,
		"task_id":       taskID,
		"agent_type":    optAgentType,
		"chain_type":    msgChainType,
	}).Info("=== PROMPT EXECUTION: Subtasks Generator Execution Started ===")

	chain := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, systemGeneratorTmpl),
		llms.TextParts(llms.ChatMessageTypeHuman, userGeneratorTmpl),
	}

	memorist, err := fp.GetMemoristHandler(ctx, &taskID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get memorist handler: %w", err)
	}

	searcher, err := fp.GetTaskSearcherHandler(ctx, taskID)
	if err != nil {
		return nil, fmt.Errorf("failed to get searcher handler: %w", err)
	}

	cfg := tools.GeneratorExecutorConfig{
		TaskID:   taskID,
		Memorist: memorist,
		Searcher: searcher,
		SubtaskList: func(ctx context.Context, name string, args json.RawMessage) (string, error) {
			err := json.Unmarshal(args, &subtaskList)
			if err != nil {
				return "", fmt.Errorf("failed to unmarshal subtask list: %w", err)
			}
			
			// LOG SUBTASK LIST TOOL CALL
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-subtasks-generator-execution",
				"action":    "subtask_list_tool_called",
				"flow":      1,
				"tool_name": name,
				"tool_args": string(args),
				"subtasks_parsed": subtaskList,
				"task_id":   taskID,
			}).Info("=== TOOL EXECUTION: Subtask List Tool Called ===")
			
			return "subtask list successfully processed", nil
		},
	}
	executor, err := fp.executor.GetGeneratorExecutor(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to get generator executor: %w", err)
	}

	chainBlob, err := json.Marshal(chain)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal msg chain: %w", err)
	}

	msgChain, err := fp.db.CreateMsgChain(ctx, database.CreateMsgChainParams{
		Type:          msgChainType,
		Model:         fp.Model(optAgentType),
		ModelProvider: string(fp.Type()),
		Chain:         chainBlob,
		FlowID:        fp.flowID,
		TaskID:        database.Int64ToNullInt64(&taskID),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create msg chain: %w", err)
	}

	// LOG MSG CHAIN CREATED
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-subtasks-generator-execution",
		"action":    "msg_chain_created",
		"flow":      1,
		"msg_chain_id": msgChain.ID,
		"chain_type":   msgChainType,
		"model":        fp.Model(optAgentType),
		"task_id":      taskID,
	}).Info("=== CHAIN CREATION: Message Chain Created for Subtasks Generator ===")

	ctx = tools.PutAgentContext(ctx, msgChainType)
	err = fp.performAgentChain(ctx, optAgentType, msgChain.ID, &taskID, nil, chain, executor, fp.summarizer)
	if err != nil {
		return nil, fmt.Errorf("failed to get subtasks generator result: %w", err)
	}

	// LOG SUBTASKS GENERATOR EXECUTION COMPLETE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-subtasks-generator-execution",
		"action":    "subtasks_generator_execution_complete",
		"flow":      1,
		"generated_subtasks": subtaskList.Subtasks,
		"subtasks_count":     len(subtaskList.Subtasks),
		"task_id":            taskID,
		"msg_chain_id":       msgChain.ID,
	}).Info("=== PROMPT EXECUTION: Subtasks Generator Execution Complete ===")

	if agentCtx, ok := tools.GetAgentContext(ctx); ok {
		fp.agentLog.PutLog(
			ctx,
			agentCtx.ParentAgentType,
			agentCtx.CurrentAgentType,
			input,
			fp.subtasksToMarkdown(subtaskList.Subtasks),
			&taskID,
			nil,
		)
	}

	return subtaskList.Subtasks, nil
}

func (fp *flowProvider) performSubtasksRefiner(
	ctx context.Context,
	taskID int64,
	systemRefinerTmpl, userRefinerTmpl, input string,
) ([]tools.SubtaskInfo, error) {
	var (
		subtaskList  tools.SubtaskList
		chain        []llms.MessageContent
		optAgentType = provider.OptionsTypeRefiner
		msgChainType = database.MsgchainTypeRefiner
	)

	// LOG SUBTASKS REFINER EXECUTION START
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-subtasks-refiner-execution",
		"action":    "subtasks_refiner_execution_start",
		"flow":      1,
		"system_prompt": systemRefinerTmpl,
		"user_prompt":   userRefinerTmpl,
		"input":         input,
		"task_id":       taskID,
		"agent_type":    optAgentType,
		"chain_type":    msgChainType,
	}).Info("=== PROMPT EXECUTION: Subtasks Refiner Execution Started ===")

	restoreChain := func(msgChain json.RawMessage) ([]llms.MessageContent, error) {
		var msgList []llms.MessageContent
		err := json.Unmarshal(msgChain, &msgList)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal chain: %w", err)
		}
		return msgList, nil
	}
	
	msgChain, err := fp.db.GetFlowTaskTypeLastMsgChain(ctx, database.GetFlowTaskTypeLastMsgChainParams{
		FlowID: fp.flowID,
		TaskID: database.Int64ToNullInt64(&taskID),
		Type:   msgChainType,
	})
	var msgChainID int64
	if err != nil {
		// Create new chain
		chain = []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeSystem, systemRefinerTmpl),
			llms.TextParts(llms.ChatMessageTypeHuman, userRefinerTmpl),
		}

		chainBlob, err := json.Marshal(chain)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal msg chain: %w", err)
		}

		msgChainRecord, err := fp.db.CreateMsgChain(ctx, database.CreateMsgChainParams{
			Type:          msgChainType,
			Model:         fp.Model(optAgentType),
			ModelProvider: string(fp.Type()),
			Chain:         chainBlob,
			FlowID:        fp.flowID,
			TaskID:        database.Int64ToNullInt64(&taskID),
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create msg chain: %w", err)
		}
		msgChainID = msgChainRecord.ID

		// LOG NEW MSG CHAIN CREATED
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-subtasks-refiner-execution",
			"action":    "msg_chain_created_new",
			"flow":      1,
			"msg_chain_id": msgChainID,
			"chain_type":   msgChainType,
			"model":        fp.Model(optAgentType),
			"task_id":      taskID,
			"initial_chain": chain,
		}).Info("=== CHAIN CREATION: New Message Chain Created for Subtasks Refiner ===")
	} else {
		// Restore existing chain
		chain, err = restoreChain(msgChain.Chain)
		if err != nil {
			return nil, fmt.Errorf("failed to restore chain: %w", err)
		}
		msgChainID = msgChain.ID

		// LOG EXISTING MSG CHAIN RESTORED
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-subtasks-refiner-execution",
			"action":    "msg_chain_restored",
			"flow":      1,
			"msg_chain_id": msgChainID,
			"chain_type":   msgChainType,
			"model":        fp.Model(optAgentType),
			"task_id":      taskID,
			"restored_chain_length": len(chain),
		}).Info("=== CHAIN RESTORATION: Existing Message Chain Restored for Subtasks Refiner ===")
	}

	memorist, err := fp.GetMemoristHandler(ctx, &taskID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get memorist handler: %w", err)
	}

	searcher, err := fp.GetTaskSearcherHandler(ctx, taskID)
	if err != nil {
		return nil, fmt.Errorf("failed to get searcher handler: %w", err)
	}

	cfg := tools.GeneratorExecutorConfig{
		TaskID:   taskID,
		Memorist: memorist,
		Searcher: searcher,
		SubtaskList: func(ctx context.Context, name string, args json.RawMessage) (string, error) {
			err := json.Unmarshal(args, &subtaskList)
			if err != nil {
				return "", fmt.Errorf("failed to unmarshal subtask list: %w", err)
			}
			
			// LOG SUBTASK LIST TOOL CALL IN REFINER
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-subtasks-refiner-execution",
				"action":    "subtask_list_tool_called_refiner",
				"flow":      1,
				"tool_name": name,
				"tool_args": string(args),
				"subtasks_refined": subtaskList,
				"task_id":   taskID,
			}).Info("=== TOOL EXECUTION: Subtask List Tool Called in Refiner ===")
			
			return "subtask list successfully processed", nil
		},
	}
	executor, err := fp.executor.GetGeneratorExecutor(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to get generator executor: %w", err)
	}

	ctx = tools.PutAgentContext(ctx, msgChainType)
	err = fp.performAgentChain(ctx, optAgentType, msgChainID, &taskID, nil, chain, executor, fp.summarizer)
	if err != nil {
		return nil, fmt.Errorf("failed to get subtasks refiner result: %w", err)
	}

	// LOG SUBTASKS REFINER EXECUTION COMPLETE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-subtasks-refiner-execution",
		"action":    "subtasks_refiner_execution_complete",
		"flow":      1,
		"refined_subtasks": subtaskList.Subtasks,
		"subtasks_count":   len(subtaskList.Subtasks),
		"task_id":          taskID,
		"msg_chain_id":     msgChainID,
	}).Info("=== PROMPT EXECUTION: Subtasks Refiner Execution Complete ===")

	if agentCtx, ok := tools.GetAgentContext(ctx); ok {
		fp.agentLog.PutLog(
			ctx,
			agentCtx.ParentAgentType,
			agentCtx.CurrentAgentType,
			input,
			fp.subtasksToMarkdown(subtaskList.Subtasks),
			&taskID,
			nil,
		)
	}

	return subtaskList.Subtasks, nil
}

func (fp *flowProvider) performCoder(
	ctx context.Context,
	taskID, subtaskID *int64,
	systemCoderTmpl, userCoderTmpl, question string,
) (string, error) {
	var (
		codeResult   tools.CodeResult
		optAgentType = provider.OptionsTypeCoder
		msgChainType = database.MsgchainTypeCoder
	)

	// LOG CODER EXECUTION START
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-coder-execution",
		"action":    "coder_execution_start",
		"flow":      1,
		"system_prompt": systemCoderTmpl,
		"user_prompt":   userCoderTmpl,
		"question":      question,
		"task_id":       taskID,
		"subtask_id":    subtaskID,
		"agent_type":    optAgentType,
		"chain_type":    msgChainType,
	}).Info("=== PROMPT EXECUTION: Coder Execution Started ===")

	adviser, err := fp.GetAskAdviceHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get adviser handler: %w", err)
	}

	installer, err := fp.GetInstallerHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get installer handler: %w", err)
	}

	memorist, err := fp.GetMemoristHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get memorist handler: %w", err)
	}

	searcher, err := fp.GetSubtaskSearcherHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get searcher handler: %w", err)
	}

	cfg := tools.CoderExecutorConfig{
		TaskID:    taskID,
		SubtaskID: subtaskID,
		Adviser:   adviser,
		Installer: installer,
		Memorist:  memorist,
		Searcher:  searcher,
		CodeResult: func(ctx context.Context, name string, args json.RawMessage) (string, error) {
			err := json.Unmarshal(args, &codeResult)
			if err != nil {
				return "", fmt.Errorf("failed to unmarshal result: %w", err)
			}
			
			// LOG CODE RESULT TOOL CALL
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-coder-execution",
				"action":    "code_result_tool_called",
				"flow":      1,
				"tool_name": name,
				"tool_args": string(args),
				"code_result": codeResult,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== TOOL EXECUTION: Code Result Tool Called ===")
			
			return "code result successfully processed", nil
		},
		Summarizer: fp.GetSummarizeResultHandler(taskID, subtaskID),
	}
	executor, err := fp.executor.GetCoderExecutor(cfg)
	if err != nil {
		return "", fmt.Errorf("failed to get coder executor: %w", err)
	}

	msgChainID, chain, err := fp.restoreChain(
		ctx, taskID, subtaskID, optAgentType, msgChainType, systemCoderTmpl, userCoderTmpl,
	)
	if err != nil {
		return "", fmt.Errorf("failed to restore chain: %w", err)
	}

	ctx = tools.PutAgentContext(ctx, msgChainType)
	err = fp.performAgentChain(ctx, optAgentType, msgChainID, taskID, subtaskID, chain, executor, fp.summarizer)
	if err != nil {
		return "", fmt.Errorf("failed to get task coder result: %w", err)
	}

	// LOG CODER EXECUTION COMPLETE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-coder-execution",
		"action":    "coder_execution_complete",
		"flow":      1,
		"final_code_result": codeResult,
		"task_id":           taskID,
		"subtask_id":        subtaskID,
		"msg_chain_id":      msgChainID,
	}).Info("=== PROMPT EXECUTION: Coder Execution Complete ===")

	if agentCtx, ok := tools.GetAgentContext(ctx); ok {
		fp.agentLog.PutLog(
			ctx,
			agentCtx.ParentAgentType,
			agentCtx.CurrentAgentType,
			question,
			codeResult.Result,
			taskID,
			subtaskID,
		)
	}

	return codeResult.Result, nil
}


func (fp *flowProvider) performInstaller(
	ctx context.Context,
	taskID, subtaskID *int64,
	systemInstallerTmpl, userInstallerTmpl, question string,
) (string, error) {
	var (
		maintenanceResult tools.MaintenanceResult
		optAgentType      = provider.OptionsTypeInstaller
		msgChainType      = database.MsgchainTypeInstaller
	)

	// LOG INSTALLER EXECUTION START
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-installer-execution",
		"action":    "installer_execution_start",
		"flow":      1,
		"system_prompt": systemInstallerTmpl,
		"user_prompt":   userInstallerTmpl,
		"question":      question,
		"task_id":       taskID,
		"subtask_id":    subtaskID,
		"agent_type":    optAgentType,
		"chain_type":    msgChainType,
	}).Info("=== PROMPT EXECUTION: Installer Execution Started ===")

	adviser, err := fp.GetAskAdviceHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get adviser handler: %w", err)
	}

	memorist, err := fp.GetMemoristHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get memorist handler: %w", err)
	}

	searcher, err := fp.GetSubtaskSearcherHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get searcher handler: %w", err)
	}

	cfg := tools.InstallerExecutorConfig{
		TaskID:    taskID,
		SubtaskID: subtaskID,
		Adviser:   adviser,
		Memorist:  memorist,
		Searcher:  searcher,
		MaintenanceResult: func(ctx context.Context, name string, args json.RawMessage) (string, error) {
			err := json.Unmarshal(args, &maintenanceResult)
			if err != nil {
				return "", fmt.Errorf("failed to unmarshal result: %w", err)
			}
			
			// LOG MAINTENANCE RESULT TOOL CALL
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-installer-execution",
				"action":    "maintenance_result_tool_called",
				"flow":      1,
				"tool_name": name,
				"tool_args": string(args),
				"maintenance_result": maintenanceResult,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== TOOL EXECUTION: Maintenance Result Tool Called ===")
			
			return "maintenance result successfully processed", nil
		},
		Summarizer: fp.GetSummarizeResultHandler(taskID, subtaskID),
	}
	executor, err := fp.executor.GetInstallerExecutor(cfg)
	if err != nil {
		return "", fmt.Errorf("failed to get installer executor: %w", err)
	}

	msgChainID, chain, err := fp.restoreChain(
		ctx, taskID, subtaskID, optAgentType, msgChainType, systemInstallerTmpl, userInstallerTmpl,
	)
	if err != nil {
		return "", fmt.Errorf("failed to restore chain: %w", err)
	}

	ctx = tools.PutAgentContext(ctx, msgChainType)
	err = fp.performAgentChain(ctx, optAgentType, msgChainID, taskID, subtaskID, chain, executor, fp.summarizer)
	if err != nil {
		return "", fmt.Errorf("failed to get task installer result: %w", err)
	}

	// LOG INSTALLER EXECUTION COMPLETE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-installer-execution",
		"action":    "installer_execution_complete",
		"flow":      1,
		"final_maintenance_result": maintenanceResult,
		"task_id":                  taskID,
		"subtask_id":               subtaskID,
		"msg_chain_id":             msgChainID,
	}).Info("=== PROMPT EXECUTION: Installer Execution Complete ===")

	if agentCtx, ok := tools.GetAgentContext(ctx); ok {
		fp.agentLog.PutLog(
			ctx,
			agentCtx.ParentAgentType,
			agentCtx.CurrentAgentType,
			question,
			maintenanceResult.Result,
			taskID,
			subtaskID,
		)
	}

	return maintenanceResult.Result, nil
}


func (fp *flowProvider) performMemorist(
	ctx context.Context,
	taskID, subtaskID *int64,
	systemMemoristTmpl, userMemoristTmpl, question string,
) (string, error) {
	var (
		memoristResult tools.MemoristResult
		optAgentType   = provider.OptionsTypeSearcher
		msgChainType   = database.MsgchainTypeMemorist
	)

	// LOG MEMORIST EXECUTION START
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-memorist-execution",
		"action":    "memorist_execution_start",
		"flow":      1,
		"system_prompt": systemMemoristTmpl,
		"user_prompt":   userMemoristTmpl,
		"question":      question,
		"task_id":       taskID,
		"subtask_id":    subtaskID,
		"agent_type":    optAgentType,
		"chain_type":    msgChainType,
	}).Info("=== PROMPT EXECUTION: Memorist Execution Started ===")

	cfg := tools.MemoristExecutorConfig{
		TaskID:    taskID,
		SubtaskID: subtaskID,
		SearchResult: func(ctx context.Context, name string, args json.RawMessage) (string, error) {
			err := json.Unmarshal(args, &memoristResult)
			if err != nil {
				return "", fmt.Errorf("failed to unmarshal result: %w", err)
			}
			
			// LOG MEMORIST SEARCH RESULT TOOL CALL
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-memorist-execution",
				"action":    "memorist_search_result_tool_called",
				"flow":      1,
				"tool_name": name,
				"tool_args": string(args),
				"memorist_result": memoristResult,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== TOOL EXECUTION: Memorist Search Result Tool Called ===")
			
			return "memorist result successfully processed", nil
		},
		Summarizer: fp.GetSummarizeResultHandler(taskID, subtaskID),
	}
	executor, err := fp.executor.GetMemoristExecutor(cfg)
	if err != nil {
		return "", fmt.Errorf("failed to get memorist executor: %w", err)
	}

	msgChainID, chain, err := fp.restoreChain(
		ctx, taskID, subtaskID, optAgentType, msgChainType, systemMemoristTmpl, userMemoristTmpl,
	)
	if err != nil {
		return "", fmt.Errorf("failed to restore chain: %w", err)
	}

	ctx = tools.PutAgentContext(ctx, msgChainType)
	err = fp.performAgentChain(ctx, optAgentType, msgChainID, taskID, subtaskID, chain, executor, fp.summarizer)
	if err != nil {
		return "", fmt.Errorf("failed to get task memorist result: %w", err)
	}

	// LOG MEMORIST EXECUTION COMPLETE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-memorist-execution",
		"action":    "memorist_execution_complete",
		"flow":      1,
		"final_memorist_result": memoristResult,
		"task_id":               taskID,
		"subtask_id":            subtaskID,
		"msg_chain_id":          msgChainID,
	}).Info("=== PROMPT EXECUTION: Memorist Execution Complete ===")

	if agentCtx, ok := tools.GetAgentContext(ctx); ok {
		fp.agentLog.PutLog(
			ctx,
			agentCtx.ParentAgentType,
			agentCtx.CurrentAgentType,
			question,
			memoristResult.Result,
			taskID,
			subtaskID,
		)
	}

	return memoristResult.Result, nil
}

func (fp *flowProvider) performPentester(
	ctx context.Context,
	taskID, subtaskID *int64,
	systemPentesterTmpl, userPentesterTmpl, question string,
) (string, error) {
	var (
		hackResult   tools.HackResult
		optAgentType = provider.OptionsTypePentester
		msgChainType = database.MsgchainTypePentester
	)

	// LOG PENTESTER EXECUTION START
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-pentester-execution",
		"action":    "pentester_execution_start",
		"flow":      1,
		"system_prompt": systemPentesterTmpl,
		"user_prompt":   userPentesterTmpl,
		"question":      question,
		"task_id":       taskID,
		"subtask_id":    subtaskID,
		"agent_type":    optAgentType,
		"chain_type":    msgChainType,
	}).Info("=== PROMPT EXECUTION: Pentester Execution Started ===")

	adviser, err := fp.GetAskAdviceHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get adviser handler: %w", err)
	}

	coder, err := fp.GetCoderHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get coder handler: %w", err)
	}

	installer, err := fp.GetInstallerHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get installer handler: %w", err)
	}

	memorist, err := fp.GetMemoristHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get memorist handler: %w", err)
	}

	searcher, err := fp.GetSubtaskSearcherHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get searcher handler: %w", err)
	}

	cfg := tools.PentesterExecutorConfig{
		TaskID:    taskID,
		SubtaskID: subtaskID,
		Adviser:   adviser,
		Coder:     coder,
		Installer: installer,
		Memorist:  memorist,
		Searcher:  searcher,
		HackResult: func(ctx context.Context, name string, args json.RawMessage) (string, error) {
			err := json.Unmarshal(args, &hackResult)
			if err != nil {
				return "", fmt.Errorf("failed to unmarshal result: %w", err)
			}
			
			// LOG HACK RESULT TOOL CALL
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-pentester-execution",
				"action":    "hack_result_tool_called",
				"flow":      1,
				"tool_name": name,
				"tool_args": string(args),
				"hack_result": hackResult,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== TOOL EXECUTION: Hack Result Tool Called ===")
			
			return "hack result successfully processed", nil
		},
		Summarizer: fp.GetSummarizeResultHandler(taskID, subtaskID),
	}
	executor, err := fp.executor.GetPentesterExecutor(cfg)
	if err != nil {
		return "", fmt.Errorf("failed to get pentester executor: %w", err)
	}

	msgChainID, chain, err := fp.restoreChain(
		ctx, taskID, subtaskID, optAgentType, msgChainType, systemPentesterTmpl, userPentesterTmpl,
	)
	if err != nil {
		return "", fmt.Errorf("failed to restore chain: %w", err)
	}

	ctx = tools.PutAgentContext(ctx, msgChainType)
	err = fp.performAgentChain(ctx, optAgentType, msgChainID, taskID, subtaskID, chain, executor, fp.summarizer)
	if err != nil {
		return "", fmt.Errorf("failed to get task pentester result: %w", err)
	}

	// LOG PENTESTER EXECUTION COMPLETE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-pentester-execution",
		"action":    "pentester_execution_complete",
		"flow":      1,
		"final_hack_result": hackResult,
		"task_id":           taskID,
		"subtask_id":        subtaskID,
		"msg_chain_id":      msgChainID,
	}).Info("=== PROMPT EXECUTION: Pentester Execution Complete ===")

	if agentCtx, ok := tools.GetAgentContext(ctx); ok {
		fp.agentLog.PutLog(
			ctx,
			agentCtx.ParentAgentType,
			agentCtx.CurrentAgentType,
			question,
			hackResult.Result,
			taskID,
			subtaskID,
		)
	}

	return hackResult.Result, nil
}



func (fp *flowProvider) performSearcher(
	ctx context.Context,
	taskID, subtaskID *int64,
	systemSearcherTmpl, userSearcherTmpl, question string,
) (string, error) {
	var (
		searchResult tools.SearchResult
		optAgentType = provider.OptionsTypeSearcher
		msgChainType = database.MsgchainTypeSearcher
	)

	// LOG SEARCHER EXECUTION START
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-searcher-execution",
		"action":    "searcher_execution_start",
		"flow":      1,
		"system_prompt": systemSearcherTmpl,
		"user_prompt":   userSearcherTmpl,
		"question":      question,
		"task_id":       taskID,
		"subtask_id":    subtaskID,
		"agent_type":    optAgentType,
		"chain_type":    msgChainType,
	}).Info("=== PROMPT EXECUTION: Searcher Execution Started ===")

	memorist, err := fp.GetMemoristHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get memorist handler: %w", err)
	}

	cfg := tools.SearcherExecutorConfig{
		TaskID:    taskID,
		SubtaskID: subtaskID,
		Memorist:  memorist,
		SearchResult: func(ctx context.Context, name string, args json.RawMessage) (string, error) {
			err := json.Unmarshal(args, &searchResult)
			if err != nil {
				return "", fmt.Errorf("failed to unmarshal result: %w", err)
			}
			
			// LOG SEARCH RESULT TOOL CALL
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-searcher-execution",
				"action":    "search_result_tool_called",
				"flow":      1,
				"tool_name": name,
				"tool_args": string(args),
				"search_result": searchResult,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== TOOL EXECUTION: Search Result Tool Called ===")
			
			return "search result successfully processed", nil
		},
		Summarizer: fp.GetSummarizeResultHandler(taskID, subtaskID),
	}
	executor, err := fp.executor.GetSearcherExecutor(cfg)
	if err != nil {
		return "", fmt.Errorf("failed to get searcher executor: %w", err)
	}

	msgChainID, chain, err := fp.restoreChain(
		ctx, taskID, subtaskID, optAgentType, msgChainType, systemSearcherTmpl, userSearcherTmpl,
	)
	if err != nil {
		return "", fmt.Errorf("failed to restore chain: %w", err)
	}

	ctx = tools.PutAgentContext(ctx, msgChainType)
	err = fp.performAgentChain(ctx, optAgentType, msgChainID, taskID, subtaskID, chain, executor, fp.summarizer)
	if err != nil {
		return "", fmt.Errorf("failed to get task searcher result: %w", err)
	}

	// LOG SEARCHER EXECUTION COMPLETE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-searcher-execution",
		"action":    "searcher_execution_complete",
		"flow":      1,
		"final_search_result": searchResult,
		"task_id":             taskID,
		"subtask_id":          subtaskID,
		"msg_chain_id":        msgChainID,
	}).Info("=== PROMPT EXECUTION: Searcher Execution Complete ===")

	if agentCtx, ok := tools.GetAgentContext(ctx); ok {
		fp.agentLog.PutLog(
			ctx,
			agentCtx.ParentAgentType,
			agentCtx.CurrentAgentType,
			question,
			searchResult.Result,
			taskID,
			subtaskID,
		)
	}

	return searchResult.Result, nil
}

func (fp *flowProvider) performEnricher(
	ctx context.Context,
	taskID, subtaskID *int64,
	systemEnricherTmpl, userEnricherTmpl, question string,
) (string, error) {
	var (
		enricherResult tools.EnricherResult
		optAgentType   = provider.OptionsTypeEnricher
		msgChainType   = database.MsgchainTypeEnricher
	)

	// LOG ENRICHER EXECUTION START
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-enricher-execution",
		"action":    "enricher_execution_start",
		"flow":      1,
		"system_prompt": systemEnricherTmpl,
		"user_prompt":   userEnricherTmpl,
		"question":      question,
		"task_id":       taskID,
		"subtask_id":    subtaskID,
		"agent_type":    optAgentType,
		"chain_type":    msgChainType,
	}).Info("=== PROMPT EXECUTION: Enricher Execution Started ===")

	memorist, err := fp.GetMemoristHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get memorist handler: %w", err)
	}

	searcher, err := fp.GetSubtaskSearcherHandler(ctx, taskID, subtaskID)
	if err != nil {
		return "", fmt.Errorf("failed to get searcher handler: %w", err)
	}

	cfg := tools.EnricherExecutorConfig{
		TaskID:    taskID,
		SubtaskID: subtaskID,
		Memorist:  memorist,
		Searcher:  searcher,
		EnricherResult: func(ctx context.Context, name string, args json.RawMessage) (string, error) {
			err := json.Unmarshal(args, &enricherResult)
			if err != nil {
				return "", fmt.Errorf("failed to unmarshal result: %w", err)
			}
			
			// LOG ENRICHER RESULT TOOL CALL
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-enricher-execution",
				"action":    "enricher_result_tool_called",
				"flow":      1,
				"tool_name": name,
				"tool_args": string(args),
				"enricher_result": enricherResult,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== TOOL EXECUTION: Enricher Result Tool Called ===")
			
			return "enrich result successfully processed", nil
		},
	}
	executor, err := fp.executor.GetEnricherExecutor(cfg)
	if err != nil {
		return "", fmt.Errorf("failed to get enricher executor: %w", err)
	}

	msgChainID, chain, err := fp.restoreChain(
		ctx, taskID, subtaskID, optAgentType, msgChainType, systemEnricherTmpl, userEnricherTmpl,
	)
	if err != nil {
		return "", fmt.Errorf("failed to restore chain: %w", err)
	}

	ctx = tools.PutAgentContext(ctx, msgChainType)
	err = fp.performAgentChain(ctx, optAgentType, msgChainID, taskID, subtaskID, chain, executor, fp.summarizer)
	if err != nil {
		return "", fmt.Errorf("failed to get task enricher result: %w", err)
	}

	// LOG ENRICHER EXECUTION COMPLETE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-enricher-execution",
		"action":    "enricher_execution_complete",
		"flow":      1,
		"final_enricher_result": enricherResult,
		"task_id":               taskID,
		"subtask_id":            subtaskID,
		"msg_chain_id":          msgChainID,
	}).Info("=== PROMPT EXECUTION: Enricher Execution Complete ===")

	if agentCtx, ok := tools.GetAgentContext(ctx); ok {
		fp.agentLog.PutLog(
			ctx,
			agentCtx.ParentAgentType,
			agentCtx.CurrentAgentType,
			question,
			enricherResult.Result,
			taskID,
			subtaskID,
		)
	}

	return enricherResult.Result, nil
}


func (fp *flowProvider) performSimpleChain(
	ctx context.Context,
	taskID, subtaskID *int64,
	opt provider.ProviderOptionsType,
	msgChainType database.MsgchainType,
	systemTmpl, userTmpl string,
) (string, error) {
	var (
		resp *llms.ContentResponse
		err  error
	)

	// LOG SIMPLE CHAIN EXECUTION START
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-simple-chain-execution",
		"action":    "simple_chain_execution_start",
		"flow":      1,
		"system_prompt": systemTmpl,
		"user_prompt":   userTmpl,
		"task_id":       taskID,
		"subtask_id":    subtaskID,
		"agent_type":    opt,
		"chain_type":    msgChainType,
	}).Info("=== PROMPT EXECUTION: Simple Chain Execution Started ===")

	chain := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, systemTmpl),
		llms.TextParts(llms.ChatMessageTypeHuman, userTmpl),
	}

	for idx := 0; idx <= maxRetriesToCallSimpleChain; idx++ {
		if idx == maxRetriesToCallSimpleChain {
			return "", fmt.Errorf("failed to call simple chain: %w", err)
		}

		// LOG SIMPLE CHAIN CALL ATTEMPT
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-simple-chain-execution",
			"action":    "simple_chain_call_attempt",
			"flow":      1,
			"attempt":   idx + 1,
			"max_retries": maxRetriesToCallSimpleChain,
			"task_id":   taskID,
			"subtask_id": subtaskID,
		}).Info("=== PROMPT EXECUTION: Simple Chain Call Attempt ===")

		resp, err = fp.CallEx(ctx, opt, chain, nil)
		if err == nil {
			// LOG SIMPLE CHAIN CALL SUCCESS
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-simple-chain-execution",
				"action":    "simple_chain_call_success",
				"flow":      1,
				"attempt":   idx + 1,
				"choices_count": len(resp.Choices),
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== PROMPT EXECUTION: Simple Chain Call Successful ===")
			break
		} else {
			// LOG SIMPLE CHAIN CALL FAILURE
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-simple-chain-execution",
				"action":    "simple_chain_call_failure",
				"flow":      1,
				"attempt":   idx + 1,
				"error":     err.Error(),
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Warn("=== PROMPT EXECUTION: Simple Chain Call Failed ===")

			if errors.Is(err, context.Canceled) {
				return "", err
			}

			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(time.Second * 5):
			default:
			}
		}
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	var parts []string
	var inputTokens, outputTokens int64
	for _, choice := range resp.Choices {
		parts = append(parts, choice.Content)
		inputTokens, outputTokens = fp.GetUsage(choice.GenerationInfo)
	}
	chain = append(chain, llms.TextParts(llms.ChatMessageTypeAI, parts...))

	result := strings.Join(parts, "\n\n")

	// LOG SIMPLE CHAIN EXECUTION COMPLETE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-simple-chain-execution",
		"action":    "simple_chain_execution_complete",
		"flow":      1,
		"result":    result,
		"input_tokens":  inputTokens,
		"output_tokens": outputTokens,
		"task_id":   taskID,
		"subtask_id": subtaskID,
		"chain_type": msgChainType,
	}).Info("=== PROMPT EXECUTION: Simple Chain Execution Complete ===")

	chainBlob, err := json.Marshal(chain)
	if err != nil {
		return "", fmt.Errorf("failed to marshal summarizer msg chain: %w", err)
	}

	_, err = fp.db.CreateMsgChain(ctx, database.CreateMsgChainParams{
		Type:          msgChainType,
		Model:         fp.Model(opt),
		ModelProvider: string(fp.Type()),
		UsageIn:       inputTokens,
		UsageOut:      outputTokens,
		Chain:         chainBlob,
		FlowID:        fp.flowID,
		TaskID:        database.Int64ToNullInt64(taskID),
		SubtaskID:     database.Int64ToNullInt64(subtaskID),
	})

	return result, nil
}
