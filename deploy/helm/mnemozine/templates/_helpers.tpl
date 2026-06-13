{{/*
Common template helpers for the mnemozine chart.
*/}}

{{- define "mnemozine.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully-qualified app name. Truncated to 63 chars for k8s name limits.
*/}}
{{- define "mnemozine.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "mnemozine.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels applied to every object.
*/}}
{{- define "mnemozine.labels" -}}
helm.sh/chart: {{ include "mnemozine.chart" . }}
{{ include "mnemozine.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{- define "mnemozine.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mnemozine.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Per-component selector labels. Pass a dict: (dict "root" $ "component" "mcp").
*/}}
{{- define "mnemozine.componentSelectorLabels" -}}
{{ include "mnemozine.selectorLabels" .root }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "mnemozine.componentLabels" -}}
{{ include "mnemozine.labels" .root }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/*
The mnemozine application image reference (mcp / ingest / maintenance share it).
*/}}
{{- define "mnemozine.image" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end -}}

{{- define "mnemozine.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "mnemozine.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Names of the shared ConfigMap and Secret.
*/}}
{{- define "mnemozine.configMapName" -}}
{{- printf "%s-config" (include "mnemozine.fullname" .) -}}
{{- end -}}

{{- define "mnemozine.secretName" -}}
{{- printf "%s-secret" (include "mnemozine.fullname" .) -}}
{{- end -}}

{{/*
Component object names.
*/}}
{{- define "mnemozine.falkordb.fullname" -}}{{ printf "%s-falkordb" (include "mnemozine.fullname" .) }}{{- end -}}
{{- define "mnemozine.ollama.fullname" -}}{{ printf "%s-ollama" (include "mnemozine.fullname" .) }}{{- end -}}
{{- define "mnemozine.qwen.fullname" -}}{{ printf "%s-qwen" (include "mnemozine.fullname" .) }}{{- end -}}
{{- define "mnemozine.litellm.fullname" -}}{{ printf "%s-litellm" (include "mnemozine.fullname" .) }}{{- end -}}
{{- define "mnemozine.mcp.fullname" -}}{{ printf "%s-mcp" (include "mnemozine.fullname" .) }}{{- end -}}
{{- define "mnemozine.ingest.fullname" -}}{{ printf "%s-ingest" (include "mnemozine.fullname" .) }}{{- end -}}
{{- define "mnemozine.maintenance.fullname" -}}{{ printf "%s-maintenance" (include "mnemozine.fullname" .) }}{{- end -}}

{{/*
Resolved backend endpoints. When a bundled dependency is enabled we use its
in-cluster Service DNS; otherwise the matching endpoints.external.* override.
*/}}
{{- define "mnemozine.falkordbUrl" -}}
{{- if .Values.falkordb.enabled -}}
redis://{{ include "mnemozine.falkordb.fullname" . }}:{{ .Values.falkordb.service.port }}
{{- else -}}
{{- required "endpoints.external.falkordbUrl is required when falkordb.enabled=false" .Values.endpoints.external.falkordbUrl -}}
{{- end -}}
{{- end -}}

{{- define "mnemozine.ollamaBaseUrl" -}}
{{- if .Values.ollama.enabled -}}
http://{{ include "mnemozine.ollama.fullname" . }}:{{ .Values.ollama.service.port }}
{{- else -}}
{{- required "endpoints.external.ollamaBaseUrl is required when ollama.enabled=false" .Values.endpoints.external.ollamaBaseUrl -}}
{{- end -}}
{{- end -}}

{{/*
Extraction LLM base_url: the in-cluster LiteLLM gateway when enabled, else the
external override.
*/}}
{{- define "mnemozine.extractionBaseUrl" -}}
{{- if .Values.litellm.enabled -}}
http://{{ include "mnemozine.litellm.fullname" . }}:{{ .Values.litellm.service.port }}/v1
{{- else -}}
{{- required "endpoints.external.extractionBaseUrl is required when litellm.enabled=false" .Values.endpoints.external.extractionBaseUrl -}}
{{- end -}}
{{- end -}}

{{/*
Upstream the LiteLLM gateway proxies to: the in-cluster qwen Service when
enabled, else litellm.upstream.apiBase.
*/}}
{{- define "mnemozine.litellmUpstreamApiBase" -}}
{{- if .Values.litellm.upstream.apiBase -}}
{{- .Values.litellm.upstream.apiBase -}}
{{- else if .Values.qwen.enabled -}}
http://{{ include "mnemozine.qwen.fullname" . }}:{{ .Values.qwen.service.port }}/v1
{{- else -}}
{{- required "Set litellm.upstream.apiBase when qwen.enabled=false" .Values.litellm.upstream.apiBase -}}
{{- end -}}
{{- end -}}
