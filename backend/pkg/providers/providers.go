package providers

import (
	"context"
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"strings"
	"sync/atomic"

	"pentagi/pkg/config"
	"pentagi/pkg/csum"
	"pentagi/pkg/database"
	"pentagi/pkg/docker"
	obs "pentagi/pkg/observability"
	"pentagi/pkg/providers/anthropic"
	"pentagi/pkg/providers/custom"
	"pentagi/pkg/providers/embeddings"
	"pentagi/pkg/providers/openai"
	"pentagi/pkg/providers/provider"
	"pentagi/pkg/templates"
	"pentagi/pkg/tools"

	"github.com/sirupsen/logrus"
)

const deltaCallCounter = 10000

const pentestDockerImage = "vxcontrol/kali-linux"

type ProviderController interface {
	NewFlowProvider(
		ctx context.Context,
		db database.Querier,
		prvtype provider.ProviderType,
		prompter templates.Prompter,
		executor tools.FlowToolsExecutor,
		flowID int64,
		input string,
	) (FlowProvider, error)
	LoadFlowProvider(
		db database.Querier,
		prvtype provider.ProviderType,
		prompter templates.Prompter,
		executor tools.FlowToolsExecutor,
		flowID int64,
		image, language, title string,
	) (FlowProvider, error)
	NewAssistantProvider(
		ctx context.Context,
		db database.Querier,
		prvtype provider.ProviderType,
		prompter templates.Prompter,
		executor tools.FlowToolsExecutor,
		assistantID, flowID int64,
		image, input string,
		streamCb StreamMessageHandler,
	) (AssistantProvider, error)
	LoadAssistantProvider(
		db database.Querier,
		prvtype provider.ProviderType,
		prompter templates.Prompter,
		executor tools.FlowToolsExecutor,
		assistantID, flowID int64,
		image, language, title string,
		streamCb StreamMessageHandler,
	) (AssistantProvider, error)
	Get(ptype provider.ProviderType) (provider.Provider, error)
	Embedder() embeddings.Embedder
	List() provider.ProvidersList
	ListStrings() []string
}

type providerController struct {
	docker   docker.DockerClient
	publicIP string
	embedder embeddings.Embedder

	startCallNumber *atomic.Int64

	defaultDockerImageForPentest string

	summarizerAgent     csum.Summarizer
	summarizerAssistant csum.Summarizer

	provider.Providers
}

func NewProviderController(cfg *config.Config, docker docker.DockerClient) (ProviderController, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is required")
	}

	embedder, err := embeddings.New(cfg)
	if err != nil {
		logrus.WithError(err).Errorf("failed to create embedder '%s'", cfg.EmbeddingProvider)
	}

	providers := make(provider.Providers)

	if cfg.OpenAIKey != "" {
		provider, err := openai.New(cfg)
		if err != nil {
			return nil, fmt.Errorf("failed to create openai provider: %w", err)
		}

		providers[provider.Type()] = provider
	}

	if cfg.LLMServerURL != "" && (cfg.LLMServerModel != "" || cfg.LLMServerConfig != "") {
		provider, err := custom.New(cfg)
		if err != nil {
			return nil, fmt.Errorf("failed to create custom provider: %w", err)
		}

		providers[provider.Type()] = provider
	}

	if cfg.AnthropicAPIKey != "" {
		provider, err := anthropic.New(cfg)
		if err != nil {
			return nil, fmt.Errorf("failed to create anthropic provider: %w", err)
		}

		providers[provider.Type()] = provider
	}

	summarizerAgent := csum.NewSummarizer(csum.SummarizerConfig{
		PreserveLast:   cfg.SummarizerPreserveLast,
		UseQA:          cfg.SummarizerUseQA,
		SummHumanInQA:  cfg.SummarizerSumHumanInQA,
		LastSecBytes:   cfg.SummarizerLastSecBytes,
		MaxBPBytes:     cfg.SummarizerMaxBPBytes,
		MaxQASections:  cfg.SummarizerMaxQASections,
		MaxQABytes:     cfg.SummarizerMaxQABytes,
		KeepQASections: cfg.SummarizerKeepQASections,
	})

	summarizerAssistant := csum.NewSummarizer(csum.SummarizerConfig{
		PreserveLast:   cfg.AssistantSummarizerPreserveLast,
		UseQA:          true,
		SummHumanInQA:  false,
		LastSecBytes:   cfg.AssistantSummarizerLastSecBytes,
		MaxBPBytes:     cfg.AssistantSummarizerMaxBPBytes,
		MaxQASections:  cfg.AssistantSummarizerMaxQASections,
		MaxQABytes:     cfg.AssistantSummarizerMaxQABytes,
		KeepQASections: cfg.AssistantSummarizerKeepQASections,
	})

	return &providerController{
		docker:   docker,
		publicIP: cfg.DockerPublicIP,
		embedder: embedder,

		startCallNumber: newAtomicInt64(0), // 0 means to make it random

		defaultDockerImageForPentest: cfg.DockerDefaultImageForPentest,

		summarizerAgent:     summarizerAgent,
		summarizerAssistant: summarizerAssistant,

		Providers: providers,
	}, nil
}

func (pc *providerController) NewFlowProvider(
	ctx context.Context,
	db database.Querier,
	prvtype provider.ProviderType,
	prompter templates.Prompter,
	executor tools.FlowToolsExecutor,
	flowID int64,
	input string,
) (FlowProvider, error) {
	ctx, span := obs.Observer.NewSpan(ctx, obs.SpanKindInternal, "providers.NewFlowProvider")
	defer span.End()

	prv, err := pc.Get(prvtype)
	if err != nil {
		return nil, fmt.Errorf("failed to get provider: %w", err)
	}

	// === IMAGE CHOOSER PROMPT ===
	imageTmpl, err := prompter.RenderTemplate(templates.PromptTypeImageChooser, map[string]any{
		"DefaultImage":           pc.docker.GetDefaultImage(),
		"DefaultImageForPentest": pc.defaultDockerImageForPentest,
		"Input":                  input,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get primary docker image template: %w", err)
	}

	// LOG IMAGE CHOOSER PROMPT
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-providers-creation",
		"action":    "image_chooser_prompt",
		"flow":      1,
		"prompt":    imageTmpl,
		"params": map[string]any{
			"DefaultImage":           pc.docker.GetDefaultImage(),
			"DefaultImageForPentest": pc.defaultDockerImageForPentest,
			"Input":                  input,
		},
		"prompt_type": templates.PromptTypeImageChooser,
	}).Info("=== PROMPT GENERATION: Image Chooser Template Rendered ===")

	image, err := prv.Call(ctx, provider.OptionsTypeSimple, imageTmpl)
	if err != nil {
		return nil, fmt.Errorf("failed to get primary docker image: %w", err)
	}
	image = strings.ToLower(strings.TrimSpace(image))

	// LOG IMAGE CHOOSER RESPONSE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-providers-creation",
		"action":    "image_chooser_response",
		"flow":      1,
		"response":  image,
		"prompt_type": templates.PromptTypeImageChooser,
	}).Info("=== PROMPT RESPONSE: Image Chooser Response Received ===")

	// === LANGUAGE CHOOSER PROMPT ===
	languageTmpl, err := prompter.RenderTemplate(templates.PromptTypeLanguageChooser, map[string]any{
		"Input": input,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get language template: %w", err)
	}

	// LOG LANGUAGE CHOOSER PROMPT
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-providers-creation",
		"action":    "language_chooser_prompt",
		"flow":      1,
		"prompt":    languageTmpl,
		"params": map[string]any{
			"Input": input,
		},
		"prompt_type": templates.PromptTypeLanguageChooser,
	}).Info("=== PROMPT GENERATION: Language Chooser Template Rendered ===")

	language, err := prv.Call(ctx, provider.OptionsTypeSimple, languageTmpl)
	if err != nil {
		return nil, fmt.Errorf("failed to get language: %w", err)
	}
	language = strings.TrimSpace(language)

	// LOG LANGUAGE CHOOSER RESPONSE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-providers-creation",
		"action":    "language_chooser_response",
		"flow":      1,
		"response":  language,
		"prompt_type": templates.PromptTypeLanguageChooser,
	}).Info("=== PROMPT RESPONSE: Language Chooser Response Received ===")

	// === FLOW DESCRIPTOR PROMPT ===
	titleTmpl, err := prompter.RenderTemplate(templates.PromptTypeFlowDescriptor, map[string]any{
		"Input":       input,
		"Lang":        language,
		"CurrentTime": getCurrentTime(),
		"N":           20,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get flow title template: %w", err)
	}

	// LOG FLOW DESCRIPTOR PROMPT
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-providers-creation",
		"action":    "flow_descriptor_prompt",
		"flow":      1,
		"prompt":    titleTmpl,
		"params": map[string]any{
			"Input":       input,
			"Lang":        language,
			"CurrentTime": getCurrentTime(),
			"N":           20,
		},
		"prompt_type": templates.PromptTypeFlowDescriptor,
	}).Info("=== PROMPT GENERATION: Flow Descriptor Template Rendered ===")

	title, err := prv.Call(ctx, provider.OptionsTypeSimple, titleTmpl)
	if err != nil {
		return nil, fmt.Errorf("failed to get flow title: %w", err)
	}
	title = strings.TrimSpace(title)

	// LOG FLOW DESCRIPTOR RESPONSE
	logrus.WithContext(ctx).WithFields(logrus.Fields{
		"type":      "MARKER",
		"component": "pentagi-providers-creation",
		"action":    "flow_descriptor_response",
		"flow":      1,
		"response":  title,
		"prompt_type": templates.PromptTypeFlowDescriptor,
	}).Info("=== PROMPT RESPONSE: Flow Descriptor Response Received ===")

	fp := &flowProvider{
		db:          db,
		embedder:    pc.embedder,
		flowID:      flowID,
		publicIP:    pc.publicIP,
		callCounter: newAtomicInt64(pc.startCallNumber.Add(deltaCallCounter)),
		image:       image,
		title:       title,
		language:    language,
		prompter:    prompter,
		executor:    executor,
		summarizer:  pc.summarizerAgent,
		Provider:    prv,
	}

	return fp, nil
}

func (pc *providerController) LoadFlowProvider(
	db database.Querier,
	prvtype provider.ProviderType,
	prompter templates.Prompter,
	executor tools.FlowToolsExecutor,
	flowID int64,
	image, language, title string,
) (FlowProvider, error) {
	prv, err := pc.Get(prvtype)
	if err != nil {
		return nil, fmt.Errorf("failed to get provider: %w", err)
	}

	fp := &flowProvider{
		db:          db,
		embedder:    pc.embedder,
		flowID:      flowID,
		publicIP:    pc.publicIP,
		callCounter: newAtomicInt64(pc.startCallNumber.Add(deltaCallCounter)),
		image:       image,
		title:       title,
		language:    language,
		prompter:    prompter,
		executor:    executor,
		summarizer:  pc.summarizerAgent,
		Provider:    prv,
	}

	return fp, nil
}

func (pc *providerController) Embedder() embeddings.Embedder {
	return pc.embedder
}

func (pc *providerController) NewAssistantProvider(
	ctx context.Context,
	db database.Querier,
	prvtype provider.ProviderType,
	prompter templates.Prompter,
	executor tools.FlowToolsExecutor,
	assistantID, flowID int64,
	image, input string,
	streamCb StreamMessageHandler,
) (AssistantProvider, error) {
	ctx, span := obs.Observer.NewSpan(ctx, obs.SpanKindInternal, "providers.NewAssistantProvider")
	defer span.End()

	prv, err := pc.Get(prvtype)
	if err != nil {
		return nil, fmt.Errorf("failed to get provider: %w", err)
	}

	languageTmpl, err := prompter.RenderTemplate(templates.PromptTypeLanguageChooser, map[string]any{
		"Input": input,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get language template: %w", err)
	}

	language, err := prv.Call(ctx, provider.OptionsTypeSimple, languageTmpl)
	if err != nil {
		return nil, fmt.Errorf("failed to get language: %w", err)
	}
	language = strings.TrimSpace(language)

	titleTmpl, err := prompter.RenderTemplate(templates.PromptTypeFlowDescriptor, map[string]any{
		"Input":       input,
		"Lang":        language,
		"CurrentTime": getCurrentTime(),
		"N":           20,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get flow title template: %w", err)
	}

	title, err := prv.Call(ctx, provider.OptionsTypeSimple, titleTmpl)
	if err != nil {
		return nil, fmt.Errorf("failed to get flow title: %w", err)
	}
	title = strings.TrimSpace(title)

	ap := &assistantProvider{
		id:         assistantID,
		summarizer: pc.summarizerAssistant,
		fp: flowProvider{
			db:          db,
			embedder:    pc.embedder,
			flowID:      flowID,
			publicIP:    pc.publicIP,
			callCounter: newAtomicInt64(pc.startCallNumber.Add(deltaCallCounter)),
			image:       image,
			title:       title,
			language:    language,
			prompter:    prompter,
			executor:    executor,
			streamCb:    streamCb,
			summarizer:  pc.summarizerAgent,
			Provider:    prv,
		},
	}

	return ap, nil
}

func (pc *providerController) LoadAssistantProvider(
	db database.Querier,
	prvtype provider.ProviderType,
	prompter templates.Prompter,
	executor tools.FlowToolsExecutor,
	assistantID, flowID int64,
	image, language, title string,
	streamCb StreamMessageHandler,
) (AssistantProvider, error) {
	prv, err := pc.Get(prvtype)
	if err != nil {
		return nil, fmt.Errorf("failed to get provider: %w", err)
	}

	ap := &assistantProvider{
		id:         assistantID,
		summarizer: pc.summarizerAssistant,
		fp: flowProvider{
			db:          db,
			embedder:    pc.embedder,
			flowID:      flowID,
			publicIP:    pc.publicIP,
			callCounter: newAtomicInt64(pc.startCallNumber.Add(deltaCallCounter)),
			image:       image,
			title:       title,
			language:    language,
			prompter:    prompter,
			executor:    executor,
			streamCb:    streamCb,
			summarizer:  pc.summarizerAgent,
			Provider:    prv,
		},
	}

	return ap, nil
}

func newAtomicInt64(seed int64) *atomic.Int64 {
	var number atomic.Int64

	if seed == 0 {
		bigID, err := rand.Int(rand.Reader, big.NewInt(math.MaxInt64))
		if err != nil {
			return &number
		}
		seed = bigID.Int64()
	}

	number.Store(seed)
	return &number
}
