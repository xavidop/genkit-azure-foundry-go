// Copyright 2025 Xavier Portilla Edo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

// Package azureaifoundry provides a comprehensive Azure AI Foundry plugin for Firebase Genkit Go.
// This plugin supports text generation and chat capabilities using Azure OpenAI and other models
// available through Azure AI Foundry.
package azureaifoundry

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/azure"
	"github.com/openai/openai-go/v3/option"
)

const provider = "azureaifoundry"

// AzureAIFoundry provides configuration options for the Azure AI Foundry plugin.
type AzureAIFoundry struct {
	Endpoint   string                 // Azure AI Foundry endpoint URL (required)
	APIKey     string                 // API key for authentication (required if not using DefaultAzureCredential)
	APIVersion string                 // Azure OpenAI API version (e.g., "2024-12-01-preview", "2024-02-01"). Defaults to "2024-12-01-preview" if not specified
	Credential azcore.TokenCredential // Optional: Use Azure DefaultAzureCredential instead of API key

	mu      sync.Mutex // Mutex to control access
	client  openai.Client
	initted bool // Whether the plugin has been initialized
}

// ModelDefinition represents a model with its name and type.
type ModelDefinition struct {
	Name           string // Model deployment name in Azure AI Foundry
	Type           string // Type: "chat", "text"
	MaxTokens      int32  // Maximum tokens the model can handle (optional)
	SupportsVision bool   // Whether the model supports vision/images (optional)
}

// Name returns the provider name.
func (a *AzureAIFoundry) Name() string {
	return provider
}

// Init initializes the Azure AI Foundry plugin.
func (a *AzureAIFoundry) Init(ctx context.Context) []api.Action {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.initted {
		panic("azureaifoundry: Init already called")
	}

	// Validate required configuration
	if a.Endpoint == "" {
		panic("azureaifoundry: Endpoint is required")
	}

	// Create client options
	var opts []option.RequestOption
	// Construct base URL by appending /openai/v1 to the endpoint
	endpoint := strings.TrimSuffix(a.Endpoint, "/")
	baseURL := fmt.Sprintf("%s/openai/v1", endpoint)
	opts = append(opts, option.WithBaseURL(baseURL))

	// Set API version (default to latest if not specified)
	if a.APIVersion != "" {
		opts = append(opts, option.WithQueryAdd("api-version", a.APIVersion))
	}

	if a.APIKey != "" {
		// Use API key authentication
		opts = append(opts, azure.WithAPIKey(a.APIKey))
	} else if a.Credential != nil {
		// Use token credential
		opts = append(opts, azure.WithTokenCredential(a.Credential))
	} else {
		// Try default Azure credential
		cred, err := azidentity.NewDefaultAzureCredential(nil)
		if err != nil {
			panic(fmt.Sprintf("azureaifoundry: failed to create default credential: %v", err))
		}
		opts = append(opts, azure.WithTokenCredential(cred))
	}

	a.client = openai.NewClient(opts...)
	a.initted = true

	return []api.Action{}
}

// DefineModel defines a model in the registry.
func (a *AzureAIFoundry) DefineModel(g *genkit.Genkit, model ModelDefinition, info *ai.ModelInfo) ai.Model {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.initted {
		panic("azureaifoundry: Init not called")
	}

	// Auto-detect model capabilities if not provided
	if info == nil {
		info = a.inferModelCapabilities(model.Name, model.Type, model.SupportsVision)
	}

	// Create model metadata
	meta := &ai.ModelOptions{
		Label:    provider + "-" + model.Name,
		Supports: info.Supports,
		Versions: info.Versions,
	}

	// Create the model function
	return genkit.DefineModel(g, api.NewName(provider, model.Name), meta, func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		return a.generateText(ctx, model.Name, input, cb)
	})
}

// DefineEmbedder defines an embedder in the registry.
func (a *AzureAIFoundry) DefineEmbedder(g *genkit.Genkit, modelName string) ai.Embedder {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.initted {
		panic("azureaifoundry: Init not called")
	}

	return genkit.DefineEmbedder(g, api.NewName(provider, modelName), nil, func(
		ctx context.Context,
		req *ai.EmbedRequest,
	) (*ai.EmbedResponse, error) {
		return a.embed(ctx, modelName, req)
	})
}

// inferModelCapabilities infers model capabilities based on model info.
func (a *AzureAIFoundry) inferModelCapabilities(modelName, modelType string, supportsVision bool) *ai.ModelInfo {
	// Detect tool support based on model name
	supportsTools := strings.Contains(strings.ToLower(modelName), "gpt-4") ||
		strings.Contains(strings.ToLower(modelName), "gpt-35-turbo") ||
		strings.Contains(strings.ToLower(modelName), "gpt-3.5-turbo")

	return &ai.ModelInfo{
		Label: modelName,
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      supportsTools,
			SystemRole: true,
			Media:      supportsVision,
		},
	}
}

// generateText handles text generation using Azure OpenAI
func (a *AzureAIFoundry) generateText(ctx context.Context, modelName string, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Build chat completion parameters
	params := a.buildChatCompletionParams(input, modelName)

	// Handle streaming vs non-streaming
	if cb != nil {
		return a.generateTextStream(ctx, params, input, cb)
	}
	return a.generateTextSync(ctx, params, input)
}

// convertMessagesToOpenAI converts Genkit messages to OpenAI message format
func (a *AzureAIFoundry) convertMessagesToOpenAI(messages []*ai.Message) []openai.ChatCompletionMessageParamUnion {
	var openAIMessages []openai.ChatCompletionMessageParamUnion

	for _, msg := range messages {
		if len(msg.Content) == 0 {
			continue // Skip messages with no content
		}

		switch msg.Role {
		case ai.RoleSystem:
			openAIMessages = append(openAIMessages, openai.ChatCompletionMessageParamUnion{
				OfSystem: &openai.ChatCompletionSystemMessageParam{
					Content: openai.ChatCompletionSystemMessageParamContentUnion{
						OfString: openai.String(msg.Content[0].Text),
					},
				},
			})
		case ai.RoleUser:
			openAIMessages = append(openAIMessages, openai.ChatCompletionMessageParamUnion{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String(msg.Content[0].Text),
					},
				},
			})
		case ai.RoleModel:
			// Extract all content parts and tool requests
			var textContent string
			var toolCalls []openai.ChatCompletionMessageToolCallUnionParam

			for _, part := range msg.Content {
				if part.IsText() {
					textContent += part.Text
				} else if part.IsToolRequest() {
					toolReq := part.ToolRequest
					// Marshal the input to JSON string
					argsJSON, err := json.Marshal(toolReq.Input)
					if err != nil {
						continue
					}
					toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
						OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
							ID:   fmt.Sprintf("call_%s", toolReq.Name),
							Type: "function",
							Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
								Name:      toolReq.Name,
								Arguments: string(argsJSON),
							},
						},
					})
				}
			}

			assistantMsg := &openai.ChatCompletionAssistantMessageParam{
				Content: openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(textContent),
				},
			}

			if len(toolCalls) > 0 {
				assistantMsg.ToolCalls = toolCalls
			}

			openAIMessages = append(openAIMessages, openai.ChatCompletionMessageParamUnion{
				OfAssistant: assistantMsg,
			})
		case ai.RoleTool:
			// Handle tool response messages
			for _, part := range msg.Content {
				if part.IsToolResponse() {
					toolResp := part.ToolResponse
					// Marshal the output to JSON string for content
					outputJSON, err := json.Marshal(toolResp.Output)
					if err != nil {
						continue
					}
					openAIMessages = append(openAIMessages, openai.ChatCompletionMessageParamUnion{
						OfTool: &openai.ChatCompletionToolMessageParam{
							Content: openai.ChatCompletionToolMessageParamContentUnion{
								OfString: openai.String(string(outputJSON)),
							},
							ToolCallID: fmt.Sprintf("call_%s", toolResp.Name),
						},
					})
				}
			}
		}
	}

	return openAIMessages
}

// extractConfig extracts and validates configuration values from a ModelRequest
type modelConfig struct {
	maxTokens   *int64
	temperature *float64
	topP        *float64
	toolChoice  string
}

// extractConfigFromRequest safely extracts configuration values from request
func (a *AzureAIFoundry) extractConfigFromRequest(input *ai.ModelRequest) *modelConfig {
	config := &modelConfig{}

	if input.Config == nil {
		return config
	}

	configMap, ok := input.Config.(map[string]interface{})
	if !ok {
		return config
	}

	if maxTokens, ok := configMap["maxOutputTokens"].(int); ok {
		val := int64(maxTokens)
		config.maxTokens = &val
	}
	if temp, ok := configMap["temperature"].(float64); ok {
		config.temperature = &temp
	}
	if topP, ok := configMap["topP"].(float64); ok {
		config.topP = &topP
	}
	if toolChoice, ok := configMap["toolChoice"].(string); ok {
		config.toolChoice = toolChoice
	}

	return config
}

// buildChatCompletionParams builds OpenAI chat completion parameters from Genkit request
func (a *AzureAIFoundry) buildChatCompletionParams(input *ai.ModelRequest, modelName string) openai.ChatCompletionNewParams {
	messages := a.convertMessagesToOpenAI(input.Messages)

	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(modelName),
		Messages: messages,
	}

	// Apply configuration if provided
	config := a.extractConfigFromRequest(input)
	if config.maxTokens != nil {
		params.MaxTokens = openai.Int(*config.maxTokens)
	}
	if config.temperature != nil {
		params.Temperature = openai.Float(*config.temperature)
	}
	if config.topP != nil {
		params.TopP = openai.Float(*config.topP)
	}

	// Handle tools
	if len(input.Tools) > 0 {
		var tools []openai.ChatCompletionToolUnionParam
		for _, tool := range input.Tools {
			// Convert Genkit tool definition to OpenAI function tool format
			funcDef := openai.FunctionDefinitionParam{
				Name: tool.Name,
			}
			if tool.Description != "" {
				funcDef.Description = openai.String(tool.Description)
			}
			if tool.InputSchema != nil {
				funcDef.Parameters = tool.InputSchema
			}
			tools = append(tools, openai.ChatCompletionFunctionTool(funcDef))
		}
		params.Tools = tools

		// Set tool choice if specified in config
		switch config.toolChoice {
		case "auto":
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String(string(openai.ChatCompletionToolChoiceOptionAutoAuto)),
			}
		case "required":
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String(string(openai.ChatCompletionToolChoiceOptionAutoRequired)),
			}
		case "none":
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String(string(openai.ChatCompletionToolChoiceOptionAutoNone)),
			}
		}
	}

	return params
}

// generateTextSync handles synchronous text generation
func (a *AzureAIFoundry) generateTextSync(ctx context.Context, params openai.ChatCompletionNewParams, originalInput *ai.ModelRequest) (*ai.ModelResponse, error) {
	resp, err := a.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("chat completion failed for model '%s': %w", params.Model, err)
	}

	return a.convertResponse(resp, originalInput), nil
}

// toolCallAccumulator holds tool call information during streaming
type toolCallAccumulator struct {
	id        string
	name      string
	arguments strings.Builder
}

// generateTextStream handles streaming text generation
func (a *AzureAIFoundry) generateTextStream(ctx context.Context, params openai.ChatCompletionNewParams, originalInput *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Note: Stream parameter is automatically set by NewStreaming
	stream := a.client.Chat.Completions.NewStreaming(ctx, params)
	defer func() {
		if err := stream.Close(); err != nil {
			// Log stream close error but don't override the main error
			_ = err
		}
	}()

	var fullText strings.Builder
	toolCallsMap := make(map[int]*toolCallAccumulator)

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta

			// Handle content streaming
			if delta.Content != "" {
				fullText.WriteString(delta.Content)

				if cb != nil {
					chunkResponse := &ai.ModelResponseChunk{
						Content: []*ai.Part{
							ai.NewTextPart(delta.Content),
						},
					}
					if err := cb(ctx, chunkResponse); err != nil {
						return nil, fmt.Errorf("streaming callback error: %w", err)
					}
				}
			}

			// Handle tool call deltas
			for _, toolCallDelta := range delta.ToolCalls {
				idx := int(toolCallDelta.Index)

				if toolCallsMap[idx] == nil {
					toolCallsMap[idx] = &toolCallAccumulator{
						id: toolCallDelta.ID,
					}
				}

				// Accumulate function name and arguments
				if toolCallDelta.Function.Name != "" {
					toolCallsMap[idx].name = toolCallDelta.Function.Name
				}
				if toolCallDelta.Function.Arguments != "" {
					toolCallsMap[idx].arguments.WriteString(toolCallDelta.Function.Arguments)
				}
			}
		}
	}

	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("stream error: %w", err)
	}

	// Build final message content
	var content []*ai.Part
	if fullText.Len() > 0 {
		content = append(content, ai.NewTextPart(fullText.String()))
	}

	// Add tool calls to content
	toolParts, err := a.convertToolCallsToParts(toolCallsMap)
	if err != nil {
		return nil, fmt.Errorf("failed to convert tool calls: %w", err)
	}
	content = append(content, toolParts...)

	return &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: content,
		},
		FinishReason: ai.FinishReasonStop,
	}, nil
}

// convertToolCallsToParts converts accumulated tool calls to AI parts
func (a *AzureAIFoundry) convertToolCallsToParts(toolCallsMap map[int]*toolCallAccumulator) ([]*ai.Part, error) {
	var parts []*ai.Part

	for _, toolCall := range toolCallsMap {
		if toolCall.name == "" {
			continue
		}

		var args map[string]interface{}
		if toolCall.arguments.Len() > 0 {
			if err := json.Unmarshal([]byte(toolCall.arguments.String()), &args); err != nil {
				return nil, fmt.Errorf("failed to unmarshal tool arguments for '%s': %w", toolCall.name, err)
			}
		}

		parts = append(parts, ai.NewToolRequestPart(&ai.ToolRequest{
			Name:  toolCall.name,
			Input: args,
		}))
	}

	return parts, nil
}

// convertResponse converts OpenAI response to Genkit format
func (a *AzureAIFoundry) convertResponse(resp *openai.ChatCompletion, originalInput *ai.ModelRequest) *ai.ModelResponse {
	if len(resp.Choices) == 0 {
		return &ai.ModelResponse{
			Message: &ai.Message{
				Role:    ai.RoleModel,
				Content: []*ai.Part{},
			},
			FinishReason: ai.FinishReasonUnknown,
		}
	}

	choice := resp.Choices[0]
	var content []*ai.Part

	if choice.Message.Content != "" {
		content = append(content, ai.NewTextPart(choice.Message.Content))
	}

	// Handle tool calls
	if len(choice.Message.ToolCalls) > 0 {
		for _, toolCall := range choice.Message.ToolCalls {
			// Handle function tool calls (most common case)
			if functionToolCall := toolCall.AsFunction(); functionToolCall.ID != "" {
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(functionToolCall.Function.Arguments), &args); err != nil {
					// If we can't parse arguments, skip this tool call
					continue
				}
				content = append(content, ai.NewToolRequestPart(&ai.ToolRequest{
					Name:  functionToolCall.Function.Name,
					Input: args,
				}))
			}
		}
	}

	finishReason := a.convertFinishReason(choice.FinishReason)

	usage := &ai.GenerationUsage{}
	if resp.Usage.PromptTokens > 0 {
		usage.InputTokens = int(resp.Usage.PromptTokens)
		usage.OutputTokens = int(resp.Usage.CompletionTokens)
		usage.TotalTokens = int(resp.Usage.TotalTokens)
	}

	return &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: content,
		},
		FinishReason: finishReason,
		Usage:        usage,
	}
}

// convertFinishReason converts OpenAI finish reason to Genkit format
func (a *AzureAIFoundry) convertFinishReason(reason string) ai.FinishReason {
	switch reason {
	case "stop":
		return ai.FinishReasonStop
	case "length":
		return ai.FinishReasonLength
	case "content_filter":
		return ai.FinishReasonBlocked
	case "tool_calls", "function_call":
		return ai.FinishReasonStop
	default:
		return ai.FinishReasonOther
	}
}

// embed handles embedding generation using Azure OpenAI
func (a *AzureAIFoundry) embed(ctx context.Context, modelName string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	var embeddings []*ai.Embedding

	// Process each document
	for _, doc := range req.Input {
		var inputText string
		// Extract text from document parts
		for _, part := range doc.Content {
			if part.IsText() {
				inputText += part.Text
			}
		}

		if inputText == "" {
			continue // Skip empty documents
		}

		// Call Azure OpenAI embeddings API
		resp, err := a.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
			Model: openai.EmbeddingModel(modelName),
			Input: openai.EmbeddingNewParamsInputUnion{
				OfString: openai.String(inputText),
			},
		})
		if err != nil {
			return nil, fmt.Errorf("embedding generation failed for model '%s': %w", modelName, err)
		}

		// Extract embeddings from response
		if len(resp.Data) > 0 {
			// Convert []float64 to []float32
			embedding := make([]float32, len(resp.Data[0].Embedding))
			for i, val := range resp.Data[0].Embedding {
				embedding[i] = float32(val)
			}

			embeddings = append(embeddings, &ai.Embedding{
				Embedding: embedding,
			})
		}
	}

	return &ai.EmbedResponse{
		Embeddings: embeddings,
	}, nil
}

// DefineCommonModels is a helper to define commonly used Azure OpenAI models
func DefineCommonModels(a *AzureAIFoundry, g *genkit.Genkit) map[string]ai.Model {
	models := make(map[string]ai.Model)
	//GPT-5 models
	models["gpt-5"] = a.DefineModel(g, ModelDefinition{
		Name:           "gpt-5",
		Type:           "chat",
		SupportsVision: true,
	}, nil)

	// GPT-5 Mini models
	models["gpt-5-mini"] = a.DefineModel(g, ModelDefinition{
		Name:           "gpt-5-mini",
		Type:           "chat",
		SupportsVision: true,
	}, nil)

	// GPT-4o models
	models["gpt-4o"] = a.DefineModel(g, ModelDefinition{
		Name:           "gpt-4o",
		Type:           "chat",
		SupportsVision: true,
	}, nil)

	models["gpt-4o-mini"] = a.DefineModel(g, ModelDefinition{
		Name:           "gpt-4o-mini",
		Type:           "chat",
		SupportsVision: true,
	}, nil)

	// GPT-4 Turbo models
	models["gpt-4-turbo"] = a.DefineModel(g, ModelDefinition{
		Name:           "gpt-4-turbo",
		Type:           "chat",
		SupportsVision: true,
	}, nil)

	// GPT-4 models
	models["gpt-4"] = a.DefineModel(g, ModelDefinition{
		Name: "gpt-4",
		Type: "chat",
	}, nil)

	// GPT-3.5 Turbo models
	models["gpt-35-turbo"] = a.DefineModel(g, ModelDefinition{
		Name: "gpt-35-turbo",
		Type: "chat",
	}, nil)

	return models
}

// DefineCommonEmbedders is a helper to define commonly used Azure OpenAI embedding models
func DefineCommonEmbedders(a *AzureAIFoundry, g *genkit.Genkit) map[string]ai.Embedder {
	embedders := make(map[string]ai.Embedder)

	// text-embedding-ada-002
	embedders["text-embedding-ada-002"] = a.DefineEmbedder(g, "text-embedding-ada-002")

	// text-embedding-3-small
	embedders["text-embedding-3-small"] = a.DefineEmbedder(g, "text-embedding-3-small")

	// text-embedding-3-large
	embedders["text-embedding-3-large"] = a.DefineEmbedder(g, "text-embedding-3-large")

	return embedders
}

// Model returns the Model with the given name.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, name))
}

// IsDefinedModel reports whether a model is defined.
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, api.NewName(provider, name)) != nil
}

// Embedder returns the Embedder with the given name.
func Embedder(g *genkit.Genkit, name string) ai.Embedder {
	return genkit.LookupEmbedder(g, api.NewName(provider, name))
}

// IsDefinedEmbedder reports whether an embedder is defined.
func IsDefinedEmbedder(g *genkit.Genkit, name string) bool {
	return genkit.LookupEmbedder(g, api.NewName(provider, name)) != nil
}
