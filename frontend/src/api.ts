export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ||
  'http://127.0.0.1:8085'

export type DebugInfo = {
  case_id?: string
  case_generation?: number
  case_boundary_status?: string
  workflow_stage?: string
  state_version?: number
  event_sequence?: number
  input_event_type?: string
  requested_route?: string
  guard_status?: 'clear' | 'warning' | 'urgent' | 'critical' | 'unknown' | string
  guard_report?: {
    guard_status?: string
    guard_checked_at?: string
    current_safety_status?: string
    pause_required?: boolean
    next_route?: string
    user_notice_markdown?: string
    risks?: Array<{
      risk_id?: string
      risk_type?: string
      level?: string
      status?: string
      trigger?: string
      missing_conditions?: string[]
    }>
    immediate_actions?: Array<{
      action_id?: string
      risk_id?: string
      priority?: number
      action?: string
      time_window?: string
    }>
  } | null
  fact_blackboard_version?: number
  fact_snapshot_version?: number
  fact_change_count?: number
  fact_conflict_count?: number
  evidence_name_inventory_version?: number
  decision_status?: string
  next_route?: string
  fact_sufficiency?: Record<string, unknown>
  question_batch?: {
    batch_id?: string
    fact_blackboard_version?: number
    markdown?: string
    questions?: Array<{
      question_id?: string
      decision_key?: string
      question_type?: string
      topic?: string
      prompt?: string
      answer_hint?: string
      decision_effects?: string[]
    }>
  }
  fact_snapshot_draft?: {
    fact_snapshot_draft_id?: string
    based_on_fact_blackboard_version?: number
    status?: string
    stale?: boolean
    markdown?: string
    unknown_fact_ids?: string[]
    conflict_group_ids?: string[]
  } | null
  pause_state?: Record<string, unknown> | null
  internal_evidence_requirements?: Array<Record<string, unknown>>
  evidence_requirement_changes?: Array<Record<string, unknown>>
  legal_model?: Record<string, unknown>
  legal_model_version?: number
  legal_model_status?: string
  relation_candidates?: Array<Record<string, unknown>>
  request_models?: Array<Record<string, unknown>>
  plan_retrieval_trace?: Record<string, unknown>
  plan_retrieval_gaps?: string[]
  proof_targets?: Array<Record<string, unknown>>
  formal_evidence_requirements?: Array<Record<string, unknown>>
  evidence_name_links?: Array<Record<string, unknown>>
  delivery_entries?: Array<Record<string, unknown>>
  plan_basis_refs?: Array<Record<string, unknown>>
  plan_basis_limitations?: string[]
  plan_change_summary?: string
  plan_audit_id?: string
  evidence_plan_request_id?: string
  previous_evidence_plan_version?: number
  evidence_plan_status?: string
  stale_dependencies?: string[]
  evidence_plan_version?: number
  evidence_collection_status?: string
  evidence_batch_id?: string
  evidence_batch_version?: number
  evidence_review_version?: number
  evidence_review_id?: string
  evidence_review_status?: string
  evidence_reviewed_at?: string
  evidence_observations?: Array<Record<string, unknown>>
  evidence_basis_refs?: Array<Record<string, unknown>>
  evidence_basis_missing?: string[]
  pending_evidence_verification?: Array<Record<string, unknown>>
  verification_round_count?: number
  new_fact_candidates_from_evidence?: Array<Record<string, unknown>>
  content_conflicts?: Array<Record<string, unknown>>
  quality_gaps?: string[]
  unclassified_materials?: Array<Record<string, unknown>>
  assessment_change_summary?: Record<string, unknown>
  evidence_review_report?: Record<string, unknown>
  solution_draft?: Record<string, unknown>
  solution_draft_status?: string
  solution_generation_id?: string
  solution_generated_at?: string
  plan_version_candidate?: string
  solution_based_on_fact_snapshot_version?: number
  solution_based_on_legal_model_version?: number
  solution_based_on_evidence_plan_version?: number
  solution_based_on_evidence_review_version?: number
  likelihood_assessment?: Record<string, unknown>
  likelihood_tier?: string
  likelihood_change?: string
  solution_change_summary?: Record<string, unknown>
  recommended_routes?: Array<Record<string, unknown>>
  alternative_routes?: Array<Record<string, unknown>>
  immediate_actions?: Array<Record<string, unknown>>
  case_tasks?: Array<Record<string, unknown>>
  document_suggestions?: Array<Record<string, unknown>>
  action_basis_refs?: Array<Record<string, unknown>>
  action_basis_gaps?: string[]
  conditional_plan?: boolean
  pending_solution_audit?: boolean
  solution_audit_status?: string
  solution_audit_id?: string
  solution_reviewed_at?: string
  solution_audit_report?: Record<string, unknown>
  published_solution?: Record<string, unknown>
  plan_version?: number
  previous_plan_version?: number
  plan_published_at?: string
  solution_version_summaries?: Array<Record<string, unknown>>
  solution_persistence_status?: string
  decision_trace_id?: string
  retrieval_summary?: Record<string, unknown>
  domain?: string
  confidence_tier?: string
  statute_hits?: string
  case_hits?: string
  graph_laws?: Array<Record<string, unknown> | string>
  graph_channels?: Array<Record<string, unknown> | string>
  fallback_guide?: {
    platform?: string
    url?: string
    search_tips?: string
  } | null
}

export type StatisticsChart = {
  type: 'bar' | 'line' | 'pie' | 'scatter' | 'heatmap' | 'table'
  title?: string
  reason?: string
  x_label?: string
  y_label?: string
  x_values?: Array<string | number>
  y_values?: Array<string | number>
  series?: Array<{ name?: string; data?: Array<unknown> }>
  echarts_option?: Record<string, unknown>
}

export type Statistics = {
  question?: string
  answer?: string
  summary?: string
  sql?: string
  columns?: string[]
  rows?: Array<Record<string, unknown>>
  chart?: StatisticsChart
  sources?: Array<{
    dataset_id?: string
    title?: string
    institution?: string
    years?: number[]
    quality_flags?: string[]
  }>
}

export type DocumentArtifact = {
  document_id?: string
  doc_type?: string
  filename?: string
  generated_docx_url?: string | null
  official_blank_url?: string | null
  source?: Record<string, unknown> | null
  official_template_match?: 'exact' | 'related' | 'none'
  official_template_note?: string | null
  missing_fields?: string[]
  expires_in_seconds?: number
}

export type ChatResult = {
  reply: string
  session_id: string
  debug?: DebugInfo | null
  statistics?: Statistics | null
  document?: DocumentArtifact | null
}

export type ChatEvent =
  | { type: 'token'; content?: string }
  | {
      type: 'done'
      session_id?: string
      debug?: DebugInfo | null
      statistics?: Statistics | null
      document?: DocumentArtifact | null
    }
  | { type: 'error'; message?: string }

export type OfficialTemplate = {
  template_id: string
  title: string
  case_type?: string
  authority_level?: string
  issuers?: string[]
  document_no?: string
  published_at?: string
  effective_at?: string
  source_page_url?: string
  source_pdf_url?: string
  source_pages?: number[]
  blank_pdf_sha256?: string
  official_blank_url?: string
}

export type HealthResponse = {
  status: 'ok' | 'degraded' | string
  dependencies?: Record<string, { ok?: boolean; error?: string }>
}

export type DocumentUploadResponse = {
  filename: string
  text: string
  preview: string
  sha256: string
  source_form: string
  truncated: boolean
  scan_warning: boolean
  evidence_block: string
  retained: boolean
  size_bytes: number
}

export type ImageUploadResponse = {
  enabled?: boolean
  message?: string
  analysis?: string
  image_sha256?: string
  image_meta?: {
    width?: number
    height?: number
    mime_type?: string
  }
  context_used?: boolean
  auto_injected?: boolean
}

export type DeleteConversationResponse = {
  deleted: boolean
  session_id: string
  warnings?: string[]
}

export type ChatRequest = {
  user_id: string
  session_id: string
  message: string
  case_id?: string
  request_id?: string
  idempotency_key?: string
  message_id?: string
  base_case_generation?: number
  base_state_version?: number
  base_fact_snapshot_version?: number
  base_evidence_plan_version?: number
  frontend_mode?: 'case' | 'qa'
  event_hint?: string
  attachments?: Array<{
    material_id?: string
    file_name?: string
    file_type?: string
    sha256?: string
    upload_status?: string
    evidence_requirement_id?: string
    evidence_batch_id?: string
  }>
  form_updates?: Array<Record<string, unknown>>
  control_action?: string
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health/deps`, {
    headers: { Accept: 'application/json' },
  })
  if (!response.ok) throw new Error(`健康检查失败（${response.status}）`)
  return (await response.json()) as HealthResponse
}

export async function getTemplates(): Promise<OfficialTemplate[]> {
  const response = await fetch(`${API_BASE}/api/v1/chat/document-templates`, {
    headers: { Accept: 'application/json' },
  })
  if (!response.ok) throw new Error(`官方模板目录加载失败（${response.status}）`)
  const payload = (await response.json()) as { templates?: OfficialTemplate[] }
  return payload.templates ?? []
}

export async function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch(`${API_BASE}/api/v1/chat/upload-document`, {
    method: 'POST',
    body: form,
  })
  if (!response.ok) {
    throw new Error(await readError(response, '文档解析失败'))
  }
  return (await response.json()) as DocumentUploadResponse
}

export async function uploadImage(
  file: File,
  userId: string,
  sessionId: string,
): Promise<ImageUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('user_id', userId)
  form.append('session_id', sessionId)
  form.append('auto_inject', 'false')
  const response = await fetch(`${API_BASE}/api/v1/chat/upload-image`, {
    method: 'POST',
    body: form,
  })
  if (!response.ok) {
    throw new Error(await readError(response, '图片分析失败'))
  }
  return (await response.json()) as ImageUploadResponse
}

export async function deleteConversation(
  userId: string,
  sessionId: string,
): Promise<DeleteConversationResponse> {
  const response = await fetch(
    `${API_BASE}/api/v1/chat/conversations/${encodeURIComponent(sessionId)}`,
    {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({ user_id: userId }),
    },
  )
  if (!response.ok) {
    throw new Error(await readError(response, '删除对话失败'))
  }
  return (await response.json()) as DeleteConversationResponse
}

export async function streamChat(
  request: ChatRequest,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(request),
    signal,
  })
  if (!response.ok) {
    throw new Error(await readError(response, '对话请求失败'))
  }
  if (!response.body) throw new Error('后端没有返回流式响应')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const consume = (block: string) => {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n')
    if (!data || data === '[DONE]') return
    let event: ChatEvent
    try {
      event = JSON.parse(data) as ChatEvent
    } catch {
      onEvent({ type: 'error', message: '后端返回了无法解析的流式事件' })
      return
    }
    onEvent(event)
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done })
    const blocks = buffer.split(/\r?\n\r?\n/)
    buffer = blocks.pop() ?? ''
    blocks.forEach(consume)
    if (done) break
  }
  if (buffer.trim()) consume(buffer)
}

async function readError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string; message?: string }
    return payload.detail || payload.message || `${fallback}（${response.status}）`
  } catch {
    return `${fallback}（${response.status}）`
  }
}

export function artifactUrl(path?: string | null): string | null {
  if (!path) return null
  if (/^https?:\/\//i.test(path)) return path
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`
}
