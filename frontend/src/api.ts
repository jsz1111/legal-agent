export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ||
  'http://127.0.0.1:8085'

export type DebugInfo = {
  case_id?: string
  case_boundary_status?: string
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

type ChatRequest = {
  user_id: string
  session_id: string
  message: string
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
