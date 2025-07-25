package providers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"pentagi/pkg/cast"
	"pentagi/pkg/csum"
	"pentagi/pkg/database"
	obs "pentagi/pkg/observability"
	"pentagi/pkg/observability/langfuse"
	"pentagi/pkg/providers/provider"
	"pentagi/pkg/templates"
	"pentagi/pkg/tools"

	"github.com/sirupsen/logrus"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const (
	maxRetriesToCallSimpleChain = 3
	maxRetriesToCallAgentChain  = 3
	maxRetriesToCallFunction    = 3
	maxReflectorCallsPerChain   = 3
	delayBetweenRetries         = 5 * time.Second
)

type callResult struct {
	streamID  int64
	funcCalls []llms.ToolCall
	info      map[string]any
	thinking  string
	content   string
}


func (fp *flowProvider) performAgentChain(
	ctx context.Context,
	optAgentType provider.ProviderOptionsType,
	chainID int64,
	taskID, subtaskID *int64,
	chain []llms.MessageContent,
	executor tools.ContextToolsExecutor,
	summarizer csum.Summarizer,
) error {
	ctx, span := obs.Observer.NewSpan(ctx, obs.SpanKindInternal, "providers.flowProvider.performAgentChain")
	defer span.End()

	var (
		wantToStop        bool
		detector          = &repeatingDetector{}
		summarizerHandler = fp.GetSummarizeResultHandler(taskID, subtaskID)
	)

	logger := logrus.WithContext(ctx).WithFields(logrus.Fields{
		"component":      "pentagi-tools-calling",
		"action":         "perform_agent_chain",
		"agent":          fp.Type(),
		"flow_id":        fp.flowID,
		"task_id":        taskID,
		"subtask_id":     subtaskID,
		"msg_chain_id":   chainID,
		"agent_type":     optAgentType,
	})
	logger.Info("=== TOOLS CALLING: Starting agent chain execution ===")

	// LOG AGENT CHAIN EXECUTION START
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-agent-chain-execution",
		"action":    "agent_chain_execution_start",
		"flow":      1,
		"initial_chain": chain,
		"chain_id":  chainID,
		"agent_type": optAgentType,
		"task_id":   taskID,
		"subtask_id": subtaskID,
		"available_tools": len(executor.Tools()),
	}).Info("=== PROMPT EXECUTION: Agent Chain Execution Started ===")

	executionContext, err := fp.getExecutionContext(ctx, taskID, subtaskID)
	if err != nil {
		logger.WithError(err).Error("failed to get execution context")
		return fmt.Errorf("failed to get execution context: %w", err)
	}

	// LOG EXECUTION CONTEXT
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-agent-chain-execution",
		"action":    "execution_context_prepared",
		"flow":      1,
		"execution_context": executionContext,
		"context_length": len(executionContext),
		"chain_id":  chainID,
		"task_id":   taskID,
		"subtask_id": subtaskID,
	}).Info("=== CONTEXT PREPARATION: Execution Context Prepared for Agent Chain ===")

	for {
		logger.WithFields(logrus.Fields{
			"chain_length":     len(chain),
			"available_tools":  len(executor.Tools()),
			"iteration":        "chain_loop",
		}).Info("=== TOOLS CALLING: Agent chain iteration ===")

		// LOG AGENT CHAIN ITERATION
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-agent-chain-execution",
			"action":    "agent_chain_iteration",
			"flow":      1,
			"current_chain": chain,
			"chain_length": len(chain),
			"available_tools": len(executor.Tools()),
			"iteration_count": "ongoing",
			"chain_id":  chainID,
			"task_id":   taskID,
			"subtask_id": subtaskID,
		}).Info("=== PROMPT EXECUTION: Agent Chain Iteration Started ===")

		result, err := fp.callWithRetries(ctx, chain, optAgentType, executor)
		if err != nil {
			logger.WithError(err).Error("failed to call agent chain")
			return err
		}

		logger.WithFields(logrus.Fields{
			"func_calls_count": len(result.funcCalls),
			"has_content":      len(result.content) > 0,
			"has_thinking":     len(result.thinking) > 0,
			"stream_id":        result.streamID,
		}).Info("=== TOOLS CALLING: LLM response received ===")

		// LOG LLM RESPONSE RECEIVED IN AGENT CHAIN
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-agent-chain-execution",
			"action":    "llm_response_received",
			"flow":      1,
			"response_content": result.content,
			"response_thinking": result.thinking,
			"tool_calls": result.funcCalls,
			"tool_calls_count": len(result.funcCalls),
			"has_content": len(result.content) > 0,
			"has_thinking": len(result.thinking) > 0,
			"stream_id": result.streamID,
			"chain_id":  chainID,
			"task_id":   taskID,
			"subtask_id": subtaskID,
		}).Info("=== PROMPT EXECUTION: LLM Response Received in Agent Chain ===")

		if err := fp.updateMsgChainUsage(ctx, chainID, result.info); err != nil {
			logger.WithError(err).Error("failed to update msg chain usage")
			return err
		}

		if len(result.funcCalls) == 0 {
			if optAgentType == provider.OptionsTypeAssistant {
				return fp.processAssistantResult(ctx, logger, chainID, chain, result, summarizer, summarizerHandler)
			} else {
				// LOG REFLECTOR ACTIVATION
				logrus.WithContext(ctx).WithFields(logrus.Fields{
					"type":      "MARKER",
					"component": "pentagi-agent-chain-execution",
					"action":    "reflector_activation",
					"flow":      1,
					"response_content": result.content,
					"human_message": fp.getLastHumanMessage(chain),
					"execution_context_preview": executionContext[:min(500, len(executionContext))],
					"chain_id":  chainID,
					"task_id":   taskID,
					"subtask_id": subtaskID,
				}).Info("=== PROMPT EXECUTION: Activating Reflector for Response Improvement ===")

				result, err = fp.performReflector(
					ctx, optAgentType, chainID, taskID, subtaskID,
					append(chain, llms.TextParts(llms.ChatMessageTypeAI, result.content)),
					fp.getLastHumanMessage(chain), result.content, executionContext, executor, 1)
				if err != nil {
					fields := make(logrus.Fields)
					if result != nil {
						fields["content"] = result.content[:min(1000, len(result.content))]
						fields["thinking"] = result.thinking[:min(1000, len(result.thinking))]
						fields["execution"] = executionContext[:min(1000, len(executionContext))]
					}
					logger.WithError(err).WithFields(fields).Error("failed to perform reflector")
					return err
				}
			}
		}

		msg := llms.MessageContent{Role: llms.ChatMessageTypeAI}
		for _, toolCall := range result.funcCalls {
			msg.Parts = append(msg.Parts, toolCall)
		}
		chain = append(chain, msg)

		if err := fp.updateMsgChain(ctx, chainID, chain); err != nil {
			logger.WithError(err).Error("failed to update msg chain")
			return err
		}

		// LOG TOOL CALLS PROCESSING START
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-agent-chain-execution",
			"action":    "tool_calls_processing_start",
			"flow":      1,
			"tool_calls": result.funcCalls,
			"tool_calls_count": len(result.funcCalls),
			"chain_id":  chainID,
			"task_id":   taskID,
			"subtask_id": subtaskID,
		}).Info("=== TOOL EXECUTION: Starting Tool Calls Processing ===")

		for idx, toolCall := range result.funcCalls {
			if toolCall.FunctionCall == nil {
				continue
			}

			funcName := toolCall.FunctionCall.Name
			
			// LOG INDIVIDUAL TOOL CALL START
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-agent-chain-execution",
				"action":    "individual_tool_call_start",
				"flow":      1,
				"tool_name": funcName,
				"tool_call_id": toolCall.ID,
				"tool_call_index": idx,
				"tool_args": toolCall.FunctionCall.Arguments,
				"chain_id":  chainID,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== TOOL EXECUTION: Individual Tool Call Started ===")

			response, err := fp.execToolCall(ctx, chainID, idx, result, detector, executor)
			if err != nil {
				logger.WithError(err).WithFields(logrus.Fields{
					"func_name": funcName,
					"func_args": toolCall.FunctionCall.Arguments,
				}).Error("failed to exec tool call")
				return err
			}

			// LOG INDIVIDUAL TOOL CALL COMPLETE
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-agent-chain-execution",
				"action":    "individual_tool_call_complete",
				"flow":      1,
				"tool_name": funcName,
				"tool_call_id": toolCall.ID,
				"tool_response": response,
				"response_length": len(response),
				"chain_id":  chainID,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== TOOL EXECUTION: Individual Tool Call Completed ===")

			// LOG BEFORE CHAIN APPEND - CRASH DEBUG
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-agent-chain-execution",
				"action":    "before_chain_append",
				"flow":      1,
				"tool_name": funcName,
				"chain_length_before": len(chain),
				"chain_id":  chainID,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== DEBUG: About to append tool response to chain ===")

			chain = append(chain, llms.MessageContent{
				Role: llms.ChatMessageTypeTool,
				Parts: []llms.ContentPart{
					llms.ToolCallResponse{
						ToolCallID: toolCall.ID,
						Name:       funcName,
						Content:    response,
					},
				},
			})

			// LOG AFTER CHAIN APPEND - CRASH DEBUG
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-agent-chain-execution",
				"action":    "after_chain_append",
				"flow":      1,
				"tool_name": funcName,
				"chain_length_after": len(chain),
				"chain_id":  chainID,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== DEBUG: Successfully appended tool response to chain ===")
			// LOG BEFORE UPDATE MSG CHAIN - CRASH DEBUG
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-agent-chain-execution",
				"action":    "before_update_msg_chain",
				"flow":      1,
				"tool_name": funcName,
				"chain_id":  chainID,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== DEBUG: About to update message chain in database ===")

			if err := fp.updateMsgChain(ctx, chainID, chain); err != nil {
				logger.WithError(err).Error("failed to update msg chain")
				return err
			}

			// LOG AFTER UPDATE MSG CHAIN - CRASH DEBUG
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-agent-chain-execution",
				"action":    "after_update_msg_chain",
				"flow":      1,
				"tool_name": funcName,
				"chain_id":  chainID,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== DEBUG: Successfully updated message chain in database ===")

			// LOG BEFORE BARRIER CHECK - CRASH DEBUG
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-agent-chain-execution",
				"action":    "before_barrier_check",
				"flow":      1,
				"tool_name": funcName,
				"chain_id":  chainID,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== DEBUG: About to check if function is barrier ===")

			if executor.IsBarrierFunction(funcName) {
				// LOG BARRIER FUNCTION DETECTED
				logrus.WithContext(ctx).WithFields(logrus.Fields{
					"type":      "MARKER",
					"component": "pentagi-agent-chain-execution",
					"action":    "barrier_function_detected",
					"flow":      1,
					"barrier_function": funcName,
					"chain_id":  chainID,
					"task_id":   taskID,
					"subtask_id": subtaskID,
				}).Info("=== TOOL EXECUTION: Barrier Function Detected - Stopping Chain ===")
				
				wantToStop = true
			}
		}

		// LOG TOOL CALLS PROCESSING COMPLETE
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-agent-chain-execution",
			"action":    "tool_calls_processing_complete",
			"flow":      1,
			"processed_tools_count": len(result.funcCalls),
			"want_to_stop": wantToStop,
			"chain_id":  chainID,
			"task_id":   taskID,
			"subtask_id": subtaskID,
		}).Info("=== TOOL EXECUTION: Tool Calls Processing Complete ===")

		if wantToStop {
			// LOG AGENT CHAIN STOPPING
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-agent-chain-execution",
				"action":    "agent_chain_stopping",
				"flow":      1,
				"final_chain": chain,
				"final_chain_length": len(chain),
				"chain_id":  chainID,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== PROMPT EXECUTION: Agent Chain Execution Stopping ===")
			
			// LOG BEFORE RETURN - CRASH DEBUG
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-agent-chain-execution",
				"action":    "before_return_from_agent_chain",
				"flow":      1,
				"chain_id":  chainID,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== DEBUG: About to return from performAgentChain ===")
			
			return nil
		}

		if summarizer != nil {
			// it returns the same chain state if error occurs
			chain, err = summarizer.SummarizeChain(ctx, summarizerHandler, chain)
			if err != nil {
				logger.WithError(err).Warn("failed to summarize chain")
			} else if err := fp.updateMsgChain(ctx, chainID, chain); err != nil {
				logger.WithError(err).Error("failed to update msg chain")
				return err
			}

			// LOG CHAIN SUMMARIZATION
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-agent-chain-execution",
				"action":    "chain_summarized",
				"flow":      1,
				"summarized_chain": chain,
				"summarized_chain_length": len(chain),
				"chain_id":  chainID,
				"task_id":   taskID,
				"subtask_id": subtaskID,
			}).Info("=== CHAIN MANAGEMENT: Chain Summarized ===")
		}
	}
}

func (fp *flowProvider) execToolCall(
	ctx context.Context,
	chainID int64,
	toolCallIDx int,
	result *callResult,
	detector *repeatingDetector,
	executor tools.ContextToolsExecutor,
) (string, error) {
	var (
		streamID int64
		thinking string
	)

	// use streamID and thinking only for first tool call to minimize content
	if toolCallIDx == 0 {
		streamID = result.streamID
		thinking = result.thinking
	}

	toolCall := result.funcCalls[toolCallIDx]
	funcName := toolCall.FunctionCall.Name
	funcArgs := json.RawMessage(toolCall.FunctionCall.Arguments)

	logger := logrus.WithContext(ctx).WithFields(logrus.Fields{
		"component":      "pentagi-tools-calling",
		"action":         "exec_tool_call",
		"agent":          fp.Type(),
		"flow_id":        fp.flowID,
		"func_name":      funcName,
		"tool_call_id":   toolCall.ID,
		"tool_call_idx":  toolCallIDx,
		"msg_chain_id":   chainID,
		"args_size":      len(funcArgs),
	})
	logger.Info("=== TOOLS CALLING: Executing tool call ===")


	ctx, observation := obs.Observer.NewObservation(ctx)
	opts := []langfuse.EventStartOption{
		langfuse.WithStartEventName(fmt.Sprintf("tool call %s", funcName)),
		langfuse.WithStartEventInput(funcArgs),
		langfuse.WithStartEventMetadata(map[string]any{
			"tool_call_id": toolCall.ID,
			"tool_name":    funcName,
		}),
	}
	
	logger.WithFields(logrus.Fields{
		"tool_args": string(funcArgs)[:min(500, len(funcArgs))],
	}).Info("=== TOOLS CALLING: Tool arguments ===")

	if detector.detect(toolCall) {
		response := fmt.Sprintf("tool call '%s' is repeating, please try another tool", funcName)

		observation.Event(append(opts,
			langfuse.WithStartEventStatus("failed"),
			langfuse.WithStartEventLevel(langfuse.ObservationLevelError),
			langfuse.WithStartEventOutput(response),
		)...)
		logger.Warn("failed to exec function: tool call is repeating")

		return response, nil
	}

	var (
		err      error
		response string
	)

	for idx := 0; idx <= maxRetriesToCallFunction; idx++ {
		if idx == maxRetriesToCallFunction {
			err = fmt.Errorf("reached max retries to call function: %w", err)
			logger.WithError(err).Error("failed to exec function")
			return "", fmt.Errorf("failed to exec function '%s': %w", funcName, err)
		}

		response, err = executor.Execute(ctx, streamID, toolCall.ID, funcName, thinking, funcArgs)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				return "", err
			}

			observation.Event(append(opts,
				langfuse.WithStartEventStatus(err.Error()),
				langfuse.WithStartEventLevel(langfuse.ObservationLevelError),
			)...)
			logger.WithError(err).Warn("failed to exec function")

			funcExecErr := err
			funcSchema, err := executor.GetToolSchema(funcName)
			if err != nil {
				logger.WithError(err).Error("failed to get tool schema")
				return "", fmt.Errorf("failed to get tool schema: %w", err)
			}

			funcArgs, err = fp.fixToolCallArgs(ctx, funcName, funcArgs, funcSchema, funcExecErr)
			if err != nil {
				logger.WithError(err).Error("failed to fix tool call args")
				return "", fmt.Errorf("failed to fix tool call args: %w", err)
			}
		} else {
			break
		}
	}

	observation.Event(append(opts,
		langfuse.WithStartEventStatus("success"),
		langfuse.WithStartEventOutput(response),
	)...)

	return response, nil
}

func (fp *flowProvider) callWithRetries(
	ctx context.Context,
	chain []llms.MessageContent,
	optAgentType provider.ProviderOptionsType,
	executor tools.ContextToolsExecutor,
) (*callResult, error) {
	var (
		err     error
		msgType = database.MsglogTypeAnswer
		parts   []string
		resp    *llms.ContentResponse
		result  callResult
	)
	logger := logrus.WithContext(ctx).WithFields(logrus.Fields{
		"component":    "pentagi-tools-calling",
		"action":       "call_with_retries",
		"agent_type":   optAgentType,
		"chain_length": len(chain),
		"max_retries":  maxRetriesToCallAgentChain,
	})
	logger.Info("=== TOOLS CALLING: LLM call with retries ===")

	// LOG COMPLETE CHAIN BEING SENT TO LLM
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-llm-call-execution",
		"action":    "llm_chain_input",
		"flow":      1,
		"chain":     chain,
		"chain_length": len(chain),
		"agent_type": optAgentType,
		"available_tools": len(executor.Tools()),
	}).Info("=== PROMPT EXECUTION: Complete Chain Being Sent to LLM ===")

	ticker := time.NewTicker(delayBetweenRetries)
	defer ticker.Stop()

	for idx := 0; idx <= maxRetriesToCallAgentChain; idx++ {
		logger.WithFields(logrus.Fields{
			"attempt": idx + 1,
			"stream_id": result.streamID,
		}).Info("=== TOOLS CALLING: LLM call attempt ===")

		// LOG LLM CALL ATTEMPT WITH DETAILED INFO
		logrus.WithContext(ctx).WithFields(logrus.Fields{
			"type":      "MARKER",
			"component": "pentagi-llm-call-execution",
			"action":    "llm_call_attempt",
			"flow":      1,
			"attempt":   idx + 1,
			"max_retries": maxRetriesToCallAgentChain,
			"agent_type": optAgentType,
			"chain_length": len(chain),
			"stream_id": result.streamID,
		}).Info("=== PROMPT EXECUTION: LLM Call Attempt Started ===")

		if idx == maxRetriesToCallAgentChain {
			msg := fmt.Sprintf("failed to call agent chain: max retries reached, %d", idx)
			return nil, fmt.Errorf(msg+": %w", err)
		}

		var streamCb streaming.Callback
		if fp.streamCb != nil {
			result.streamID = fp.callCounter.Add(1)
			streamCb = func(ctx context.Context, chunk streaming.Chunk) error {
				switch chunk.Type {
				case streaming.ChunkTypeReasoning:
					// LOG REASONING CHUNK
					logrus.WithContext(ctx).WithFields(logrus.Fields{
						"type":      "MARKER",
						"component": "pentagi-llm-streaming",
						"action":    "reasoning_chunk_received",
						"flow":      1,
						"stream_id": result.streamID,
						"reasoning_content": chunk.ReasoningContent,
					}).Debug("=== STREAMING: Reasoning Chunk Received ===")
					
					return fp.streamCb(ctx, &StreamMessageChunk{
						Type:     StreamMessageChunkTypeThinking,
						MsgType:  msgType,
						Thinking: chunk.ReasoningContent,
						StreamID: result.streamID,
					})
				case streaming.ChunkTypeText:
					// LOG TEXT CHUNK
					logrus.WithContext(ctx).WithFields(logrus.Fields{
						"type":      "MARKER",
						"component": "pentagi-llm-streaming",
						"action":    "text_chunk_received",
						"flow":      1,
						"stream_id": result.streamID,
						"text_content": chunk.Content,
					}).Debug("=== STREAMING: Text Chunk Received ===")
					
					return fp.streamCb(ctx, &StreamMessageChunk{
						Type:     StreamMessageChunkTypeContent,
						MsgType:  msgType,
						Content:  chunk.Content,
						StreamID: result.streamID,
					})
				case streaming.ChunkTypeToolCall:
					// LOG TOOL CALL CHUNK
					logrus.WithContext(ctx).WithFields(logrus.Fields{
						"type":      "MARKER",
						"component": "pentagi-llm-streaming",
						"action":    "tool_call_chunk_received",
						"flow":      1,
						"stream_id": result.streamID,
					}).Debug("=== STREAMING: Tool Call Chunk Received (Skipped) ===")
					// skip tool call chunks (we don't need them for now)
				case streaming.ChunkTypeDone:
					// LOG DONE CHUNK
					logrus.WithContext(ctx).WithFields(logrus.Fields{
						"type":      "MARKER",
						"component": "pentagi-llm-streaming",
						"action":    "done_chunk_received",
						"flow":      1,
						"stream_id": result.streamID,
					}).Debug("=== STREAMING: Done Chunk Received ===")
					
					return fp.streamCb(ctx, &StreamMessageChunk{
						Type:     StreamMessageChunkTypeFlush,
						MsgType:  msgType,
						StreamID: result.streamID,
					})
				}
				return nil
			}
		}

		resp, err = fp.CallWithTools(ctx, optAgentType, chain, executor.Tools(), streamCb)
		if err == nil {
			logger.WithFields(logrus.Fields{
				"attempts":        idx + 1,
				"choices_count":   len(resp.Choices),
				"tool_calls":      len(result.funcCalls),
				"content_length":  len(result.content),
				"thinking_length": len(result.thinking),
			}).Info("=== TOOLS CALLING: LLM call successful ===")

			// LOG LLM CALL SUCCESS WITH FULL RESPONSE
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-llm-call-execution",
				"action":    "llm_call_success",
				"flow":      1,
				"attempt":   idx + 1,
				"choices_count": len(resp.Choices),
				"full_response": resp,
				"agent_type": optAgentType,
			}).Info("=== PROMPT EXECUTION: LLM Call Successful ===")
			break
		} else {
			logger.WithFields(logrus.Fields{
				"attempt": idx + 1,
				"error":   err.Error(),
			}).Warn("=== TOOLS CALLING: LLM call failed, retrying ===")

			// LOG LLM CALL FAILURE
			logrus.WithContext(ctx).WithFields(logrus.Fields{
				"type":      "MARKER",
				"component": "pentagi-llm-call-execution",
				"action":    "llm_call_failure",
				"flow":      1,
				"attempt":   idx + 1,
				"error":     err.Error(),
				"agent_type": optAgentType,
			}).Warn("=== PROMPT EXECUTION: LLM Call Failed ===")
		}

		ticker.Reset(delayBetweenRetries)
		select {
		case <-ticker.C:
		case <-ctx.Done():
			return nil, fmt.Errorf("context canceled while waiting for retry: %w", ctx.Err())
		}
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	// Process response choices
	for _, choice := range resp.Choices {
		if strings.TrimSpace(choice.Content) != "" {
			parts = append(parts, choice.Content)
		}

		if choice.GenerationInfo != nil {
			result.info = choice.GenerationInfo
		}

		for _, toolCall := range choice.ToolCalls {
			if toolCall.FunctionCall == nil {
				continue
			}
			result.funcCalls = append(result.funcCalls, toolCall)
		}

		if choice.ReasoningContent != "" {
			result.thinking = choice.ReasoningContent
		}
	}

	result.content = strings.Join(parts, "\n\n")

	// LOG COMPLETE LLM RESPONSE PROCESSED
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-llm-call-execution",
		"action":    "llm_response_processed",
		"flow":      1,
		"content":   result.content,
		"thinking":  result.thinking,
		"tool_calls": result.funcCalls,
		"tool_calls_count": len(result.funcCalls),
		"agent_type": optAgentType,
		"usage_info": result.info,
	}).Info("=== PROMPT EXECUTION: LLM Response Fully Processed ===")

	return &result, nil
}


func (fp *flowProvider) performReflector(
	ctx context.Context,
	optOriginType provider.ProviderOptionsType,
	chainID int64,
	taskID, subtaskID *int64,
	chain []llms.MessageContent,
	humanMessage, content, executionContext string,
	executor tools.ContextToolsExecutor,
	iteration int,
) (*callResult, error) {
	if iteration > maxReflectorCallsPerChain {
		return &callResult{content: content}, nil
	}

	optAgentType := provider.OptionsTypeReflector
	msgChainType := database.MsgchainTypeReflector

	reflectorContext := map[string]map[string]any{
		"user": {
			"Question": humanMessage,
			"Content":  content,
		},
		"system": {
			"ExecutionContext": executionContext,
			"CurrentTime":      getCurrentTime(),
		},
	}

	// LOG REFLECTOR EXECUTION START
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-reflector-execution",
		"action":    "reflector_execution_start",
		"flow":      1,
		"iteration": iteration,
		"max_iterations": maxReflectorCallsPerChain,
		"human_message": humanMessage,
		"content_to_reflect": content,
		"execution_context_preview": executionContext[:min(500, len(executionContext))],
		"reflector_context": reflectorContext,
		"chain_id":  chainID,
		"task_id":   taskID,
		"subtask_id": subtaskID,
	}).Info("=== PROMPT EXECUTION: Reflector Execution Started ===")

	ctx, observation := obs.Observer.NewObservation(ctx)
	reflectorSpan := observation.Span(
		langfuse.WithStartSpanName("reflector agent"),
		langfuse.WithStartSpanInput(content),
		langfuse.WithStartSpanMetadata(langfuse.Metadata{
			"user_context":   reflectorContext["user"],
			"system_context": reflectorContext["system"],
		}),
	)
	ctx, _ = reflectorSpan.Observation(ctx)

	// === REFLECTOR USER PROMPT ===
	userReflectorTmpl, err := fp.prompter.RenderTemplate(templates.PromptTypeQuestionReflector, reflectorContext["user"])
	if err != nil {
		return nil, wrapErrorEndSpan(ctx, reflectorSpan, "failed to get user reflector template", err)
	}

	// LOG REFLECTOR USER PROMPT
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-reflector-execution",
		"action":    "reflector_user_prompt",
		"flow":      1,
		"prompt":    userReflectorTmpl,
		"params":    reflectorContext["user"],
		"prompt_type": templates.PromptTypeQuestionReflector,
		"iteration": iteration,
		"chain_id":  chainID,
		"task_id":   taskID,
		"subtask_id": subtaskID,
	}).Info("=== PROMPT GENERATION: Reflector User Template Rendered ===")

	// === REFLECTOR SYSTEM PROMPT ===
	systemReflectorTmpl, err := fp.prompter.RenderTemplate(templates.PromptTypeReflector, reflectorContext["system"])
	if err != nil {
		return nil, wrapErrorEndSpan(ctx, reflectorSpan, "failed to get system reflector template", err)
	}

	// LOG REFLECTOR SYSTEM PROMPT
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-reflector-execution",
		"action":    "reflector_system_prompt",
		"flow":      1,
		"prompt":    systemReflectorTmpl,
		"params":    reflectorContext["system"],
		"prompt_type": templates.PromptTypeReflector,
		"iteration": iteration,
		"chain_id":  chainID,
		"task_id":   taskID,
		"subtask_id": subtaskID,
	}).Info("=== PROMPT GENERATION: Reflector System Template Rendered ===")

	advice, err := fp.performSimpleChain(ctx, taskID, subtaskID, optAgentType,
		msgChainType, systemReflectorTmpl, userReflectorTmpl)
	if err != nil {
		advice = ToolPlaceholder
	}

	// LOG REFLECTOR ADVICE GENERATED
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-reflector-execution",
		"action":    "reflector_advice_generated",
		"flow":      1,
		"advice":    advice,
		"iteration": iteration,
		"chain_id":  chainID,
		"task_id":   taskID,
		"subtask_id": subtaskID,
	}).Info("=== PROMPT EXECUTION: Reflector Advice Generated ===")

	opts := []langfuse.SpanEndOption{
		langfuse.WithEndSpanStatus("failed"),
		langfuse.WithEndSpanOutput(advice),
		langfuse.WithEndSpanLevel(langfuse.ObservationLevelWarning),
	}
	defer reflectorSpan.End(opts...)

	chain = append(chain, llms.TextParts(llms.ChatMessageTypeHuman, advice))
	
	// LOG REFLECTOR CHAIN UPDATED
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-reflector-execution",
		"action":    "reflector_chain_updated",
		"flow":      1,
		"updated_chain": chain,
		"advice_added": advice,
		"iteration": iteration,
		"chain_id":  chainID,
		"task_id":   taskID,
		"subtask_id": subtaskID,
	}).Info("=== CHAIN MANAGEMENT: Reflector Chain Updated with Advice ===")

	result, err := fp.callWithRetries(ctx, chain, optOriginType, executor)
	if err != nil {
		logrus.WithError(err).Error("failed to call agent chain by reflector")
		opts = append(opts, langfuse.WithEndSpanStatus(err.Error()))
		return nil, err
	}

	if err := fp.updateMsgChainUsage(ctx, chainID, result.info); err != nil {
		logrus.WithError(err).Error("failed to update msg chain usage")
		return nil, err
	}

	chain = append(chain, llms.TextParts(llms.ChatMessageTypeAI, result.content))
	
	// LOG REFLECTOR RESULT
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-reflector-execution",
		"action":    "reflector_result",
		"flow":      1,
		"result_content": result.content,
		"result_thinking": result.thinking,
		"result_tool_calls": result.funcCalls,
		"tool_calls_count": len(result.funcCalls),
		"iteration": iteration,
		"chain_id":  chainID,
		"task_id":   taskID,
		"subtask_id": subtaskID,
	}).Info("=== PROMPT EXECUTION: Reflector Result Received ===")

	if len(result.funcCalls) == 0 {
		return fp.performReflector(ctx, optOriginType, chainID, taskID, subtaskID, chain,
			humanMessage, result.content, executionContext, executor, iteration+1)
	}

	opts = append(opts, langfuse.WithEndSpanStatus("success"))
	return result, nil
}

func (fp *flowProvider) getLastHumanMessage(chain []llms.MessageContent) string {
	ast, err := cast.NewChainAST(chain, true)
	if err != nil {
		return ""
	}

	slices.Reverse(ast.Sections)
	for _, section := range ast.Sections {
		if section.Header.HumanMessage != nil {
			var hparts []string
			for _, part := range section.Header.HumanMessage.Parts {
				if text, ok := part.(llms.TextContent); ok {
					hparts = append(hparts, text.Text)
				}
			}
			return strings.Join(hparts, "\n")
		}
	}

	return ""
}

func (fp *flowProvider) processAssistantResult(
	ctx context.Context,
	logger *logrus.Entry,
	chainID int64,
	chain []llms.MessageContent,
	result *callResult,
	summarizer csum.Summarizer,
	summarizerHandler tools.SummarizeHandler,
) error {
	var err error

	if fp.streamCb != nil {
		if result.streamID == 0 {
			result.streamID = fp.callCounter.Add(1)
		}
		err := fp.streamCb(ctx, &StreamMessageChunk{
			Type:     StreamMessageChunkTypeUpdate,
			MsgType:  database.MsglogTypeAnswer,
			Content:  result.content,
			Thinking: result.thinking,
			StreamID: result.streamID,
		})
		if err != nil {
			return fmt.Errorf("failed to stream assistant result: %w", err)
		}
	}

	if summarizer != nil {
		// it returns the same chain state if error occurs
		chain, err = summarizer.SummarizeChain(ctx, summarizerHandler, chain)
		if err != nil {
			logger.WithError(err).Warn("failed to summarize chain")
		}
	}

	chain = append(chain, llms.TextParts(llms.ChatMessageTypeAI, result.content))
	if err := fp.updateMsgChain(ctx, chainID, chain); err != nil {
		return fmt.Errorf("failed to update msg chain: %w", err)
	}

	return nil
}

func (fp *flowProvider) updateMsgChain(ctx context.Context, chainID int64, chain []llms.MessageContent) error {
	chainBlob, err := json.Marshal(chain)
	if err != nil {
		return fmt.Errorf("failed to marshal msg chain: %w", err)
	}

	_, err = fp.db.UpdateMsgChain(ctx, database.UpdateMsgChainParams{
		Chain: chainBlob,
		ID:    chainID,
	})
	if err != nil {
		return fmt.Errorf("failed to update msg chain in DB: %w", err)
	}

	return nil
}

func (fp *flowProvider) updateMsgChainUsage(ctx context.Context, chainID int64, info map[string]any) error {
	inputTokens, outputTokens := int64(0), int64(0)
	if info != nil {
		inputTokens, outputTokens = fp.GetUsage(info)
	}

	_, err := fp.db.UpdateMsgChainUsage(ctx, database.UpdateMsgChainUsageParams{
		UsageIn:  inputTokens,
		UsageOut: outputTokens,
		ID:       chainID,
	})
	if err != nil {
		return fmt.Errorf("failed to update msg chain usage in DB: %w", err)
	}

	return nil
}
