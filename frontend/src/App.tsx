import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlertTriangle,
  ArrowUp,
  BarChart3,
  BookOpen,
  Check,
  ChevronDown,
  CircleHelp,
  ClipboardList,
  Clock3,
  Download,
  FileText,
  FolderOpen,
  HeartHandshake,
  ImagePlus,
  Landmark,
  Link2,
  LoaderCircle,
  MessageSquarePlus,
  Paperclip,
  PanelRight,
  Plus,
  RefreshCw,
  Scale,
  Search,
  ShieldCheck,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie as ChartPie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import {
  DebugInfo,
  DocumentArtifact,
  HealthResponse,
  OfficialTemplate,
  Statistics,
  artifactUrl,
  deleteConversation,
  getHealth,
  getTemplates,
  streamChat,
  uploadDocument,
  uploadImage,
} from './api'
import './styles.css'

type Role = 'user' | 'assistant'
type WorkspaceMode = 'qa' | 'case'
type InspectorTab = 'case' | 'sources' | 'documents'

type ChatMessage = {
  id: string
  role: Role
  content: string
  attachments?: string[]
  createdAt: number
}

type AttachmentItem = {
  id: string
  file: File
  status: 'ready' | 'processing' | 'done' | 'error'
  note?: string
}

type IntakeState = {
  overview: string
  counterparty: string
  timePlaceAmount: string
  goal: string
  priorActions: string
}

type WorkspaceSnapshot = {
  sessionId: string
  messages: ChatMessage[]
  draft: string
  attachments: AttachmentItem[]
  intake: IntakeState
  debug: DebugInfo | null
  statistics: Statistics | null
  document: DocumentArtifact | null
  activeTab: InspectorTab
}

type ConversationRecord = Omit<WorkspaceSnapshot, 'attachments'> & {
  mode: WorkspaceMode
  title: string
  updatedAt: number
}

const DOMAIN_LABELS: Record<string, string> = {
  labor_social_security: '劳动和社会保障',
  consumer_market: '消费维权',
  contracts_property_housing: '合同、房产和租赁',
  criminal_public_security: '刑事和治安',
  family_vulnerable_groups: '婚姻家庭与弱势群体',
  traffic_personal_injury: '交通事故与人身损害',
  medical_education_tax: '医疗、教育和税务',
  administrative_remedies: '行政救济',
  intellectual_property: '知识产权',
  environment_pollution: '环境污染',
  cyber_data_fraud: '网络、数据和诈骗',
  mediation_notary_arbitration: '调解、公证和仲裁',
}

const DEPENDENCY_LABELS: Record<string, string> = {
  postgres: '案件资料',
  redis: '会话状态',
  minio: '文件服务',
  milvus: '法律检索',
  neo4j: '法律图谱',
  backend: '对话服务',
}

const CASE_SCENARIOS = [
  { label: '劳动纠纷', text: '公司已经3个月没发工资了，我有劳动合同、工资流水和考勤记录' },
  { label: '消费维权', text: '我在某平台买了一件商品，收到后发现是假货，有订单截图和聊天记录，商家拒绝退款' },
  { label: '租房押金', text: '退房后房东以房屋有损坏为由不退押金，但损坏不是我造成的，我有交房时的照片' },
  { label: '安全优先', text: '我现在正在遭受家庭暴力，对方威胁我不让报警' },
]

const QA_SCENARIOS = [
  { label: '法条查询', text: '《劳动合同法》对试用期有哪些限制？' },
  { label: '流程比较', text: '劳动仲裁和劳动诉讼有什么区别？分别需要多长时间？' },
  { label: '办事渠道', text: '遇到网络消费纠纷可以向哪些部门投诉？' },
  { label: '类案参考', text: '租房押金不退的类似案例通常如何处理？' },
  { label: '统计趋势', text: '2018到2020年劳动争议一审收案变化趋势？' },
]

const INITIAL_INTAKE: IntakeState = {
  overview: '',
  counterparty: '',
  timePlaceAmount: '',
  goal: '',
  priorActions: '',
}

const CONVERSATIONS_KEY = 'legal-agent:conversations:v1'
const MODE_KEY = 'legal-agent:workspace-mode'

const uid = () =>
  globalThis.crypto?.randomUUID?.() ||
  `${Date.now()}-${Math.random().toString(16).slice(2)}`

const readOrCreate = (key: string) => {
  const value = window.localStorage.getItem(key)
  if (value) return value
  const next = uid()
  window.localStorage.setItem(key, next)
  return next
}

const readModeSession = (mode: WorkspaceMode) => {
  const key = `legal-agent:session-id:${mode}`
  const existing = window.localStorage.getItem(key)
  if (existing) return existing
  if (mode === 'case') {
    const legacy = window.localStorage.getItem('legal-agent:session-id')
    if (legacy) {
      window.localStorage.setItem(key, legacy)
      return legacy
    }
  }
  return readOrCreate(key)
}

const defaultTitle = (mode: WorkspaceMode) =>
  mode === 'case' ? '新的维权案件' : '新的法律问答'

const deriveConversationTitle = (
  messages: ChatMessage[],
  mode: WorkspaceMode,
) => {
  const firstUserMessage = messages.find((message) => message.role === 'user')
  if (!firstUserMessage) return defaultTitle(mode)
  const cleaned = firstUserMessage.content
    .replace(/【[^】]+】/g, ' ')
    .replace(/以下内容由用户一次性提交[^。]*。?/g, ' ')
    .replace(/已提交材料，请结合附件继续梳理。/g, '附件材料')
    .replace(/\s+/g, ' ')
    .trim()
  if (!cleaned) return defaultTitle(mode)
  return cleaned.length > 24 ? `${cleaned.slice(0, 24)}…` : cleaned
}

const createConversation = (
  mode: WorkspaceMode,
  sessionId: string = uid(),
): ConversationRecord => ({
  mode,
  sessionId,
  title: defaultTitle(mode),
  updatedAt: Date.now(),
  messages: [],
  draft: '',
  intake: INITIAL_INTAKE,
  debug: null,
  statistics: null,
  document: null,
  activeTab: 'case',
})

const loadConversations = (): ConversationRecord[] => {
  try {
    const raw = window.localStorage.getItem(CONVERSATIONS_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as ConversationRecord[]
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter((item) =>
        item
        && (item.mode === 'qa' || item.mode === 'case')
        && typeof item.sessionId === 'string'
        && Array.isArray(item.messages),
      )
  } catch {
    return []
  }
}

const saveConversations = (items: ConversationRecord[]) => {
  const bounded = [...items]
    .sort((left, right) => right.updatedAt - left.updatedAt)
  try {
    window.localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(bounded))
  } catch {
    // A full localStorage must not interrupt the active conversation.
  }
}

const formatConversationTime = (timestamp: number) => {
  const date = new Date(timestamp)
  const today = new Date()
  if (date.toDateString() === today.toDateString()) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
  return date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
}

const formatBytes = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function MarkdownReply({ content }: { content: string }) {
  const normalized = content
    .replace(/^[ \t]*\*\*【([^】\n]{2,40})】\*\*[ \t]*$/gm, '## $1')
    .replace(/^[ \t]*【([^】\n]{2,40})】[ \t]*$/gm, '## $1')
    .replace(/^[ \t]*□[ \t]*(.+?)[：:][ \t]*$/gm, '### $1')
    .replace(/^[ \t]*·[ \t]+/gm, '- ')
    .replace(/^[ \t]*(路径[一二三四五六七八九十]+：.+)$/gm, '### $1')
    .replace(/^[ \t]*\*\*📊[ \t]*(.+)\*\*[ \t]*$/gm, '> **当前判断：** $1')
    .replace(/^[ \t]*📊[ \t]*(.+)$/gm, '> **当前判断：** $1')
    .replace(/^[ \t]*📄[ \t]*(.+)$/gm, '> **参考文书：** $1')
    .replace(/^[ \t]*🔄[ \t]*(.+)$/gm, '> **继续完善：** $1')
    .replace(
      /(^|[\s（(])(www\.[a-z0-9.-]+\.[a-z]{2,})(?=[）)，。；、\s]|$)/gim,
      '$1[$2](https://$2)',
    )
    .replace(/\n{3,}/g, '\n\n')

  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ node: _node, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer" />
          ),
        }}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  )
}

function ConfidenceBadge({ tier }: { tier?: string }) {
  const normalized = tier || 'GATHERING'
  const label =
    normalized === 'HIGH'
      ? '依据较充分'
      : normalized === 'MEDIUM'
        ? '信息基本清楚'
        : normalized === 'LOW'
          ? '仍需核验'
          : '正在梳理'
  return <span className={`confidence-badge ${normalized.toLowerCase()}`}>{label}</span>
}

function MessageCard({ message }: { message: ChatMessage }) {
  return (
    <article className={`message-card ${message.role}`}>
      <div className="message-avatar" aria-hidden="true">
        {message.role === 'assistant' ? <Scale size={17} /> : <span>我</span>}
      </div>
      <div className="message-body">
        <div className="message-meta">
          <strong>{message.role === 'assistant' ? '法护通' : '您'}</strong>
          <time>
            {new Date(message.createdAt).toLocaleTimeString('zh-CN', {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </time>
        </div>
        {message.attachments?.length ? (
          <div className="message-attachments">
            {message.attachments.map((name) => (
              <span className="attachment-pill" key={name}>
                <Paperclip size={13} />
                {name}
              </span>
            ))}
          </div>
        ) : null}
        {message.content ? (
          message.role === 'assistant' ? (
            <MarkdownReply content={message.content} />
          ) : (
            <p className="user-message-text">{message.content}</p>
          )
        ) : (
          <div className="typing-line">
            <span />
            <span />
            <span />
            正在梳理
          </div>
        )}
      </div>
    </article>
  )
}

function WelcomeState({
  mode,
  scenarios,
  onScenario,
}: {
  mode: WorkspaceMode
  scenarios: typeof CASE_SCENARIOS
  onScenario: (text: string) => void
}) {
  const isQa = mode === 'qa'
  return (
    <div className="welcome-state">
      <div className="welcome-mark">
        {isQa ? <BookOpen size={25} /> : <Scale size={26} />}
      </div>
      <p className="eyebrow">LEGAL WORKSPACE</p>
      <h1>
        {isQa ? <>先问清规则，<em>再决定怎么做。</em></> : <>把事情说清楚，<em>下一步更有底。</em></>}
      </h1>
      <p className="welcome-copy">
        {isQa
          ? '适合查询法律规定、办理流程、维权渠道、类案和法律统计。回答会尽量标明检索依据，具体纠纷请切换到案件维权。'
          : '围绕一个具体纠纷持续梳理事实、证据、期限和诉求，只在真正影响责任、管辖或行动路径时继续追问。'}
      </p>
      <div className="welcome-actions">
        {scenarios.slice(0, 5).map((scenario) => (
          <button key={scenario.label} className="scenario-chip" onClick={() => onScenario(scenario.text)}>
            {scenario.label}
            <ArrowUp size={14} />
          </button>
        ))}
      </div>
      {isQa ? (
        <div className="welcome-safety neutral">
          <BookOpen size={16} />
          <span>回答用于法律信息参考，重要决定前请核验原文或咨询专业人士。</span>
        </div>
      ) : (
        <div className="welcome-safety">
          <ShieldCheck size={17} />
          <span>涉及现实危险时，请优先拨打 110；法律援助可拨打 12348。</span>
        </div>
      )}
    </div>
  )
}

function HealthCard({
  health,
  loading,
  onRefresh,
}: {
  health: HealthResponse | null
  loading: boolean
  onRefresh: () => void
}) {
  const deps = Object.entries(health?.dependencies ?? {})
  return (
    <section className="side-section health-card">
      <div className="side-section-head">
        <div>
          <p className="section-kicker">系统状态</p>
          <h2>
            <span className={`health-dot ${health?.status === 'ok' ? 'ok' : health ? 'degraded' : ''}`} />
            {health?.status === 'ok' ? '服务正常' : health ? '部分依赖异常' : '尚未检查'}
          </h2>
        </div>
        <button className="icon-button subtle" title="刷新系统状态" onClick={onRefresh} disabled={loading}>
          <RefreshCw size={15} className={loading ? 'spin' : ''} />
        </button>
      </div>
      {deps.length > 0 ? (
        <div className="dependency-list">
          {deps.map(([name, item]) => (
            <span key={name} className={item.ok ? 'dependency ok' : 'dependency fail'}>
              <span className="dependency-dot" />
              {DEPENDENCY_LABELS[name] || name}
            </span>
          ))}
        </div>
      ) : (
        <p className="muted-copy">点击刷新查看对话、案件资料、法律检索与文件能力。</p>
      )}
    </section>
  )
}

function ConversationManager({
  mode,
  conversations,
  activeSessionId,
  onOpen,
  onCreate,
  onDelete,
}: {
  mode: WorkspaceMode
  conversations: ConversationRecord[]
  activeSessionId: string
  onOpen: (record: ConversationRecord) => void
  onCreate: () => void
  onDelete: (record: ConversationRecord) => void
}) {
  const visible = conversations
    .filter((record) => record.mode === mode)
    .sort((left, right) => right.updatedAt - left.updatedAt)
  return (
    <section className="side-section conversation-manager">
      <div className="side-section-head">
        <div>
          <p className="section-kicker">HISTORY</p>
          <h2>{mode === 'case' ? '我的案件' : '我的问答'}</h2>
        </div>
        <button
          className="icon-button subtle"
          title={mode === 'case' ? '新建案件' : '新建问答'}
          onClick={onCreate}
        >
          <Plus size={15} />
        </button>
      </div>
      <div className="conversation-records">
        {visible.map((record) => {
          const active = record.sessionId === activeSessionId
          return (
            <div className={`conversation-record ${active ? 'active' : ''}`} key={record.sessionId}>
              <button
                className="conversation-record-main"
                onClick={() => onOpen(record)}
                aria-current={active ? 'page' : undefined}
              >
                <span className="record-icon">
                  {mode === 'case' ? <FolderOpen size={14} /> : <BookOpen size={14} />}
                </span>
                <span className="record-copy">
                  <strong title={record.title}>{record.title}</strong>
                  <small>
                    <Clock3 size={11} />
                    {formatConversationTime(record.updatedAt)}
                    <span className="record-state">可继续</span>
                  </small>
                </span>
              </button>
              <button
                className="record-delete"
                title={`删除${record.title}`}
                onClick={() => onDelete(record)}
              >
                <Trash2 size={13} />
              </button>
            </div>
          )
        })}
      </div>
      {visible.length === 0 ? (
        <p className="conversation-empty">还没有记录，开始一次新对话。</p>
      ) : null}
    </section>
  )
}

function IntakePanel({
  intake,
  setIntake,
  onSubmit,
  busy,
}: {
  intake: IntakeState
  setIntake: (value: IntakeState) => void
  onSubmit: () => void
  busy: boolean
}) {
  const update = (key: keyof IntakeState, value: string) =>
    setIntake({ ...intake, [key]: value })
  return (
    <details className="intake-panel">
      <summary>
        <span className="summary-left">
          <ClipboardList size={18} />
          <span>
            <strong>首次案件材料包</strong>
            <small>一次说清可减少追问，不确定的内容可以留空</small>
          </span>
        </span>
        <ChevronDown size={17} className="summary-chevron" />
      </summary>
      <div className="intake-form">
        <label className="field wide">
          <span>事情经过</span>
          <textarea
            value={intake.overview}
            onChange={(event) => update('overview', event.target.value)}
            placeholder="按时间顺序说明发生了什么、对方做了什么、现在是什么状态"
            rows={2}
          />
        </label>
        <label className="field">
          <span>对方及双方关系</span>
          <input
            value={intake.counterparty}
            onChange={(event) => update('counterparty', event.target.value)}
            placeholder="个人、公司、商家、用人单位等"
          />
        </label>
        <label className="field">
          <span>时间、地点和金额</span>
          <input
            value={intake.timePlaceAmount}
            onChange={(event) => update('timePlaceAmount', event.target.value)}
            placeholder="大概时间、平台或地点、金额"
          />
        </label>
        <label className="field">
          <span>希望解决的结果</span>
          <input
            value={intake.goal}
            onChange={(event) => update('goal', event.target.value)}
            placeholder="退款、赔偿、履行合同、投诉等"
          />
        </label>
        <label className="field">
          <span>已经沟通或处理的情况</span>
          <input
            value={intake.priorActions}
            onChange={(event) => update('priorActions', event.target.value)}
            placeholder="协商、报警、投诉、仲裁、诉讼等"
          />
        </label>
        <div className="intake-footer">
          <span className="muted-copy">上传材料可在下方输入框旁添加，系统会保留文件指纹和核验边界。</span>
          <button className="primary-button small" onClick={onSubmit} disabled={busy}>
            <Sparkles size={15} />
            {busy ? '正在分析' : '提交材料包'}
          </button>
        </div>
      </div>
    </details>
  )
}

function StatsView({ statistics }: { statistics: Statistics }) {
  const chart = statistics.chart
  const xValues = chart?.x_values ?? []
  const series = chart?.series ?? []
  const chartRows = xValues.map((label, index) => {
    const row: Record<string, string | number> = { label: String(label) }
    series.forEach((item, seriesIndex) => {
      const value = item.data?.[index]
      if (typeof value === 'number' || typeof value === 'string') {
        row[`series_${seriesIndex}`] = value
      }
    })
    return row
  })
  const colors = ['#177e89', '#d26a45', '#5b6fa6', '#c28a36']
  const isAxisChart = chart?.type === 'line' || chart?.type === 'bar'
  const pieRows = xValues.map((label, index) => ({
    name: String(label),
    value: Number(series[0]?.data?.[index] ?? 0),
  }))
  const scatterRows = (series[0]?.data ?? []).flatMap((point) => {
    if (!Array.isArray(point) || point.length < 2) return []
    const x = Number(point[0])
    const y = Number(point[1])
    if (!Number.isFinite(x) || !Number.isFinite(y)) return []
    return [{ x, y, label: String(point[2] ?? '') }]
  })
  const heatmapPoints = (series[0]?.data ?? []).flatMap((point) => {
    if (!Array.isArray(point) || point.length < 3) return []
    const x = Number(point[0])
    const y = Number(point[1])
    const value = Number(point[2])
    if (![x, y, value].every(Number.isFinite)) return []
    return [{ x, y, value }]
  })
  const heatValues = heatmapPoints.map((point) => point.value)
  const heatMin = heatValues.length ? Math.min(...heatValues) : 0
  const heatMax = heatValues.length ? Math.max(...heatValues) : 0
  const heatmapValue = (x: number, y: number) =>
    heatmapPoints.find((point) => point.x === x && point.y === y)?.value
  const heatColor = (value: number) => {
    const ratio = heatMax === heatMin ? 0.65 : (value - heatMin) / (heatMax - heatMin)
    return `rgba(23, 126, 137, ${0.12 + ratio * 0.76})`
  }
  return (
    <div className="stats-view">
      <div className="stats-answer">
        <div className="section-kicker">统计回答</div>
        <MarkdownReply content={statistics.answer || '已返回统计结果。'} />
      </div>
      {statistics.summary ? <p className="stats-summary">{statistics.summary}</p> : null}
      {chart && chart.type !== 'table' && isAxisChart ? (
        <div className="chart-frame">
          <div className="chart-title">{chart.title || '法律统计分析'}</div>
          <ResponsiveContainer width="100%" height={230}>
            {chart.type === 'line' ? (
              <LineChart data={chartRows}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7ecef" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                {series.map((item, index) => (
                  <Line
                    key={item.name || index}
                    type="monotone"
                    dataKey={`series_${index}`}
                    name={item.name || '数值'}
                    stroke={colors[index % colors.length]}
                    strokeWidth={2.5}
                    dot={{ r: 3 }}
                  />
                ))}
              </LineChart>
            ) : (
              <BarChart data={chartRows}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7ecef" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                {series.map((item, index) => (
                  <Bar
                    key={item.name || index}
                    dataKey={`series_${index}`}
                    name={item.name || '数值'}
                    fill={colors[index % colors.length]}
                    radius={[3, 3, 0, 0]}
                  />
                ))}
              </BarChart>
            )}
          </ResponsiveContainer>
          {chart.reason ? <p className="chart-reason">{chart.reason}</p> : null}
        </div>
      ) : chart?.type === 'pie' ? (
        <div className="chart-frame">
          <div className="chart-title">{chart.title || '构成分析'}</div>
          <ResponsiveContainer width="100%" height={230}>
            <PieChart>
              <ChartPie
                data={pieRows}
                dataKey="value"
                nameKey="name"
                innerRadius={55}
                outerRadius={82}
                paddingAngle={2}
              >
                {pieRows.map((row, index) => (
                  <Cell key={row.name} fill={colors[index % colors.length]} />
                ))}
              </ChartPie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
          {chart.reason ? <p className="chart-reason">{chart.reason}</p> : null}
        </div>
      ) : chart?.type === 'scatter' && scatterRows.length ? (
        <div className="chart-frame">
          <div className="chart-title">{chart.title || '指标关系分析'}</div>
          <ResponsiveContainer width="100%" height={230}>
            <ScatterChart margin={{ top: 12, right: 18, bottom: 18, left: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e7ecef" />
              <XAxis
                type="number"
                dataKey="x"
                name={chart.x_label || '指标一'}
                tick={{ fontSize: 11 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name={chart.y_label || '指标二'}
                tick={{ fontSize: 11 }}
              />
              <ZAxis dataKey="label" name="类别" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name={series[0]?.name || '统计值'} data={scatterRows} fill="#177e89" />
            </ScatterChart>
          </ResponsiveContainer>
          {chart.reason ? <p className="chart-reason">{chart.reason}</p> : null}
        </div>
      ) : chart?.type === 'heatmap' && heatmapPoints.length ? (
        <div className="chart-frame">
          <div className="chart-title">{chart.title || '年度与类别分布'}</div>
          <div
            className="heatmap-grid"
            style={{ gridTemplateColumns: `minmax(88px, 1.2fr) repeat(${xValues.length}, minmax(58px, 1fr))` }}
          >
            <span className="heatmap-corner">{chart.y_label || '类别'}</span>
            {xValues.map((label) => <strong key={`x-${label}`}>{String(label)}</strong>)}
            {(chart.y_values ?? []).flatMap((label, yIndex) => [
              <strong className="heatmap-row-label" key={`y-${label}`}>{String(label)}</strong>,
              ...xValues.map((_xLabel, xIndex) => {
                const value = heatmapValue(xIndex, yIndex)
                return (
                  <span
                    className="heatmap-cell"
                    key={`${xIndex}-${yIndex}`}
                    style={value === undefined ? undefined : { background: heatColor(value) }}
                    title={value === undefined ? '无数据' : String(value)}
                  >
                    {value === undefined ? '—' : value.toLocaleString('zh-CN')}
                  </span>
                )
              }),
            ])}
          </div>
          {chart.reason ? <p className="chart-reason">{chart.reason}</p> : null}
        </div>
      ) : null}
      <div className="stats-table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              {(statistics.columns?.length ? statistics.columns : Object.keys(statistics.rows?.[0] ?? {})).map(
                (column) => <th key={column}>{column}</th>,
              )}
            </tr>
          </thead>
          <tbody>
            {(statistics.rows ?? []).slice(0, 100).map((row, index) => (
              <tr key={index}>
                {(statistics.columns?.length ? statistics.columns : Object.keys(row)).map((column) => (
                  <td key={column}>{String(row[column] ?? '—')}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="stats-sources">
        <div className="section-kicker">数据口径</div>
        {(statistics.sources ?? []).map((source) => (
          <div className="source-row" key={`${source.dataset_id}-${source.title}`}>
            <BarChart3 size={14} />
            <span>{source.title || '未命名统计表'} · {source.institution || '未注明机构'}</span>
            <small>{source.years?.join('、') || '年份未注明'}</small>
          </div>
        ))}
      </div>
      {statistics.sql ? (
        <details className="sql-disclosure">
          <summary>查看已通过安全校验的 SQL</summary>
          <pre>{statistics.sql}</pre>
        </details>
      ) : null}
    </div>
  )
}

function Inspector({
  mode,
  open,
  onClose,
  activeTab,
  setActiveTab,
  debug,
  statistics,
  document,
  templates,
  selectedTemplate,
  setSelectedTemplate,
  templateError,
}: {
  mode: WorkspaceMode
  open: boolean
  onClose: () => void
  activeTab: 'case' | 'sources' | 'documents'
  setActiveTab: (tab: 'case' | 'sources' | 'documents') => void
  debug: DebugInfo | null
  statistics: Statistics | null
  document: DocumentArtifact | null
  templates: OfficialTemplate[]
  selectedTemplate: string
  setSelectedTemplate: (value: string) => void
  templateError: string
}) {
  const domain = debug?.domain
    ? DOMAIN_LABELS[debug.domain] || debug.domain
    : mode === 'case'
      ? '待识别'
      : '法律知识问答'
  const template = templates.find((item) => item.template_id === selectedTemplate)
  const tabLabels: Array<[InspectorTab, string, typeof ClipboardList]> = [
    ['case', mode === 'case' ? '案情台账' : '问答概览', ClipboardList],
    ['sources', '检索依据', Search],
    ['documents', '文书模板', FileText],
  ]
  return (
    <aside className={`inspector ${open ? 'compact-open' : ''}`}>
      <div className="inspector-compact-head">
        <strong>案件资料</strong>
        <button className="icon-button subtle" title="关闭案件资料" onClick={onClose}>
          <X size={16} />
        </button>
      </div>
      <div className="inspector-tabs" role="tablist" aria-label="案件工作台">
        {tabLabels.map(([key, label, Icon]) => (
          <button
            key={key as string}
            role="tab"
            aria-selected={activeTab === key}
            className={activeTab === key ? 'active' : ''}
            onClick={() => setActiveTab(key)}
          >
            <Icon size={15} />
            {label as string}
          </button>
        ))}
      </div>
      <div className="inspector-content">
        {activeTab === 'case' ? (
          <div className="panel-stack">
            <section className="inspector-card case-overview-card">
              <div className="card-heading">
                <div className="card-icon teal"><FolderOpen size={17} /></div>
                <div>
                  <p className="section-kicker">CURRENT CASE</p>
                  <h2>{domain}</h2>
                </div>
              </div>
              <div className="case-meta-row">
                <span>工作流状态</span>
                <strong>
                  {!debug
                    ? mode === 'case' ? '等待开始' : '等待提问'
                    : debug.case_boundary_status === 'awaiting_confirmation'
                      ? '等待案件归属确认'
                      : mode === 'case' ? '案件处理中' : '检索已更新'}
                </strong>
              </div>
              <div className="case-meta-row">
                <span>信息充分度</span>
                <ConfidenceBadge tier={debug?.confidence_tier} />
              </div>
            </section>
            <section className="inspector-card">
              <div className="card-title-row">
                <h3><Paperclip size={15} />{mode === 'case' ? '材料接收边界' : '回答依据边界'}</h3>
                <span className="counter">{mode === 'case' ? '仅本案' : '本轮'}</span>
              </div>
              <p className="muted-copy">
                {mode === 'case'
                  ? '上传材料只作为待核验证据进入流程，系统不会把文件内容自动当成您已经确认的事实。'
                  : '回答优先使用本轮检索结果；没有可靠命中时应明确说明，统计资料不作为个案法律结论。'}
              </p>
              <div className="boundary-list">
                {mode === 'case' ? (
                  <>
                    <span><Check size={14} />原文件指纹</span>
                    <span><Check size={14} />来源形式</span>
                    <span><CircleHelp size={14} />真实性待核验</span>
                  </>
                ) : (
                  <>
                    <span><Check size={14} />法条原文</span>
                    <span><Check size={14} />来源口径</span>
                    <span><CircleHelp size={14} />重要决定需复核</span>
                  </>
                )}
              </div>
            </section>
            {statistics ? (
              <section className="inspector-card">
                <div className="card-title-row">
                  <h3><BarChart3 size={15} />本轮统计结果</h3>
                  <button className="text-button" onClick={() => setActiveTab('sources')}>查看</button>
                </div>
                <p className="muted-copy">{statistics.summary || statistics.answer || '统计结果已生成。'}</p>
              </section>
            ) : null}
          </div>
        ) : activeTab === 'sources' ? (
          <div className="panel-stack">
            {statistics ? (
              <section className="inspector-card stats-card">
                <StatsView statistics={statistics} />
              </section>
            ) : null}
            <section className="inspector-card source-card">
              <div className="card-title-row">
                <h3><BookOpen size={15} />法条命中</h3>
                <span className="counter">本轮</span>
              </div>
              {debug?.statute_hits ? (
                <details open>
                  <summary>查看检索原文</summary>
                  <MarkdownReply content={debug.statute_hits} />
                </details>
              ) : <p className="empty-panel">生成方案后，这里会显示本轮法条原文。</p>}
            </section>
            <section className="inspector-card source-card">
              <div className="card-title-row">
                <h3><Landmark size={15} />类案与图谱</h3>
              </div>
              {debug?.case_hits ? <MarkdownReply content={debug.case_hits} /> : <p className="empty-panel">暂无类案命中。</p>}
              {(debug?.graph_laws?.length || debug?.graph_channels?.length) ? (
                <div className="graph-list">
                  {[...(debug.graph_laws ?? []), ...(debug.graph_channels ?? [])].map((item, index) => (
                    <span key={index}>{typeof item === 'string' ? item : JSON.stringify(item)}</span>
                  ))}
                </div>
              ) : null}
              {debug?.fallback_guide ? (
                <a className="external-link" href={debug.fallback_guide.url} target="_blank" rel="noreferrer">
                  <Link2 size={14} />
                  {debug.fallback_guide.platform || '前往裁判文书网检索'}
                </a>
              ) : null}
            </section>
          </div>
        ) : (
          <div className="panel-stack">
            <section className="inspector-card document-card">
              <div className="card-title-row">
                <h3><FileText size={15} />当前生成结果</h3>
                {document ? <span className="counter">短期有效</span> : null}
              </div>
              {document ? (
                <>
                  <h4>{document.doc_type || '参考文书'}</h4>
                  <p className="muted-copy">{document.filename || '可编辑 DOCX 参考稿已生成。'}</p>
                  <div className="download-stack">
                    {artifactUrl(document.generated_docx_url) ? (
                      <a className="download-button primary" href={artifactUrl(document.generated_docx_url) ?? undefined} download>
                        <Download size={15} />下载智能填写 DOCX
                      </a>
                    ) : null}
                    {artifactUrl(document.official_blank_url) ? (
                      <a className="download-button" href={artifactUrl(document.official_blank_url) ?? undefined} download>
                        <FileText size={15} />下载相关官方空白 PDF
                      </a>
                    ) : null}
                  </div>
                  {document.official_template_note ? <p className="note-callout">{document.official_template_note}</p> : null}
                  {document.missing_fields?.length ? (
                    <div className="missing-fields">
                      <span>提交前需补充</span>
                      <p>{document.missing_fields.join('、')}</p>
                    </div>
                  ) : null}
                </>
              ) : <p className="empty-panel">在对话中回复“生成文书”，这里会出现可编辑参考稿。</p>}
            </section>
            <section className="inspector-card template-card">
              <div className="card-title-row">
                <h3><Landmark size={15} />官方空白模板库</h3>
                <span className="counter">{templates.length} 份</span>
              </div>
              <label className="select-field">
                <span>选择模板</span>
                <select value={selectedTemplate} onChange={(event) => setSelectedTemplate(event.target.value)}>
                  <option value="">请选择</option>
                  {templates.map((item) => (
                    <option value={item.template_id} key={item.template_id}>{item.title}</option>
                  ))}
                </select>
              </label>
              {templateError ? <p className="error-copy">{templateError}</p> : null}
              {template ? (
                <div className="template-source">
                  <h4>{template.title}</h4>
                  <p>{(template.issuers ?? []).join('、') || '发布机关未注明'}</p>
                  <dl>
                    <div><dt>文号</dt><dd>{template.document_no || '未注明'}</dd></div>
                    <div><dt>生效</dt><dd>{template.effective_at || '未注明'}</dd></div>
                    <div><dt>页码</dt><dd>{template.source_pages?.join(' - ') || '未注明'}</dd></div>
                  </dl>
                  {artifactUrl(template.official_blank_url) ? (
                    <a className="external-link" href={artifactUrl(template.official_blank_url) ?? undefined} download>
                      <Download size={14} />
                      下载此官方空白模板
                    </a>
                  ) : null}
                  {template.source_page_url ? (
                    <a className="external-link" href={template.source_page_url} target="_blank" rel="noreferrer">
                      <Link2 size={14} />
                      查看发布机关原文
                    </a>
                  ) : null}
                </div>
              ) : <p className="empty-panel">选择模板查看来源、文号和下载链接。</p>}
            </section>
          </div>
        )}
      </div>
    </aside>
  )
}

function App() {
  const bootRef = useRef<{
    mode: WorkspaceMode
    conversations: ConversationRecord[]
    active: ConversationRecord
  } | null>(null)
  if (!bootRef.current) {
    const storedMode = window.localStorage.getItem(MODE_KEY)
    const bootMode: WorkspaceMode = storedMode === 'qa' ? 'qa' : 'case'
    const storedConversations = loadConversations()
    const activeId = window.localStorage.getItem(`legal-agent:active-conversation:${bootMode}`)
    const storedActive = storedConversations.find(
      (record) => record.mode === bootMode && record.sessionId === activeId,
    )
    const recentActive = storedConversations
      .filter((record) => record.mode === bootMode)
      .sort((left, right) => right.updatedAt - left.updatedAt)[0]
    const active = storedActive
      || recentActive
      || createConversation(bootMode, readModeSession(bootMode))
    const conversations = storedConversations.some(
      (record) => record.sessionId === active.sessionId,
    )
      ? storedConversations
      : [active, ...storedConversations]
    bootRef.current = { mode: bootMode, conversations, active }
  }
  const boot = bootRef.current
  const [userId] = useState(() => readOrCreate('legal-agent:user-id'))
  const [mode, setMode] = useState<WorkspaceMode>(boot.mode)
  const [conversations, setConversations] = useState<ConversationRecord[]>(boot.conversations)
  const [sessionId, setSessionId] = useState(boot.active.sessionId)
  const [messages, setMessages] = useState<ChatMessage[]>(boot.active.messages)
  const [draft, setDraft] = useState(boot.active.draft)
  const [attachments, setAttachments] = useState<AttachmentItem[]>([])
  const [intake, setIntake] = useState<IntakeState>(boot.active.intake)
  const [busy, setBusy] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthLoading, setHealthLoading] = useState(false)
  const [debug, setDebug] = useState<DebugInfo | null>(boot.active.debug)
  const [statistics, setStatistics] = useState<Statistics | null>(boot.active.statistics)
  const [document, setDocument] = useState<DocumentArtifact | null>(boot.active.document)
  const [templates, setTemplates] = useState<OfficialTemplate[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState('')
  const [templateError, setTemplateError] = useState('')
  const [activeTab, setActiveTab] = useState<InspectorTab>(boot.active.activeTab)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textAreaRef = useRef<HTMLTextAreaElement>(null)
  const conversationEndRef = useRef<HTMLDivElement>(null)

  const refreshHealth = useCallback(async () => {
    setHealthLoading(true)
    try {
      setHealth(await getHealth())
    } catch (caught) {
      setHealth({ status: 'degraded', dependencies: { backend: { ok: false, error: String(caught) } } })
    } finally {
      setHealthLoading(false)
    }
  }, [])

  useEffect(() => {
    void refreshHealth()
    getTemplates()
      .then((items) => {
        setTemplates(items)
        if (items[0]) setSelectedTemplate(items[0].template_id)
      })
      .catch((caught) => setTemplateError(String(caught)))
  }, [refreshHealth])

  useEffect(() => {
    window.localStorage.setItem(MODE_KEY, mode)
    window.localStorage.setItem(`legal-agent:session-id:${mode}`, sessionId)
    window.localStorage.setItem(`legal-agent:active-conversation:${mode}`, sessionId)
  }, [mode, sessionId])

  useEffect(() => {
    setConversations((current) => {
      const existing = current.find((record) => record.sessionId === sessionId)
      const lastMessageAt = messages.at(-1)?.createdAt
      const next: ConversationRecord = {
        mode,
        sessionId,
        title: deriveConversationTitle(messages, mode),
        updatedAt: lastMessageAt || existing?.updatedAt || Date.now(),
        messages,
        draft,
        intake,
        debug,
        statistics,
        document,
        activeTab,
      }
      const updated = [
        next,
        ...current.filter((record) => record.sessionId !== sessionId),
      ]
      saveConversations(updated)
      return updated
    })
  }, [
    activeTab,
    debug,
    document,
    draft,
    intake,
    messages,
    mode,
    sessionId,
    statistics,
  ])

  useEffect(() => {
    saveConversations(conversations)
  }, [conversations])

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({
      behavior: messages.length > 2 ? 'smooth' : 'auto',
      block: 'end',
    })
  }, [messages])

  const focusComposer = () => {
    textAreaRef.current?.focus()
  }

  const handleFileSelection = (files: FileList | null) => {
    if (!files) return
    const next = Array.from(files)
      .slice(0, 8)
      .map((file) => ({
        id: uid(),
        file,
        status: 'ready' as const,
      }))
    setAttachments((current) => [...current, ...next].slice(0, 8))
    if (fileInputRef.current) fileInputRef.current.value = ''
    focusComposer()
  }

  const removeAttachment = (id: string) =>
    setAttachments((current) => current.filter((item) => item.id !== id))

  const buildIntakeMessage = () => {
    const sections = [
      ['事情经过', intake.overview],
      ['对方及双方关系', intake.counterparty],
      ['时间、地点和金额', intake.timePlaceAmount],
      ['希望解决的结果', intake.goal],
      ['已经沟通或处理的情况', intake.priorActions],
    ].filter(([, value]) => value.trim())
    if (!sections.length) return ''
    return [
      '【首次案件材料包】',
      '以下内容由用户一次性提交；未填写的项目表示本轮未提供，不能推测。',
      '',
      ...sections.flatMap(([label, value]) => [`【${label}】`, value, '']),
    ].join('\n').trim()
  }

  const send = async (messageText: string, files = attachments) => {
    const trimmed = messageText.trim()
    if (busy || (!trimmed && files.length === 0)) return
    setBusy(true)
    setError('')
    const visibleFiles = files.map((item) => item.file.name)
    const userMessageId = uid()
    const assistantMessageId = uid()
    setMessages((current) => [
      ...current,
      {
        id: userMessageId,
        role: 'user',
        content: trimmed || '已提交材料，请结合附件继续梳理。',
        attachments: visibleFiles,
        createdAt: Date.now(),
      },
      { id: assistantMessageId, role: 'assistant', content: '', createdAt: Date.now() },
    ])
    setDraft('')
    setAttachments((current) =>
      current.map((item) =>
        files.some((queued) => queued.id === item.id)
          ? { ...item, status: 'processing' }
          : item,
      ),
    )
    setUploading(files.length > 0)

    const evidenceBlocks: string[] = []
    try {
      for (const item of files.slice(0, 8)) {
        const file = item.file
        const lower = file.name.toLowerCase()
        setAttachments((current) =>
          current.map((candidate) => candidate.id === item.id ? { ...candidate, status: 'processing' } : candidate),
        )
        if (/\.(pdf|docx|txt)$/i.test(lower)) {
          const result = await uploadDocument(file)
          evidenceBlocks.push(result.evidence_block)
          setMessages((current) =>
            current.map((candidate) =>
              candidate.id === userMessageId
                ? { ...candidate, content: `${candidate.content}\n\n已提取 ${result.filename} 的可读文字，内容将按待核验证据处理。` }
                : candidate,
            ),
          )
        } else if (/^image\//.test(file.type) || /\.(png|jpe?g|gif|bmp|webp)$/i.test(lower)) {
          const result = await uploadImage(file, userId, sessionId)
          if (!result.enabled) throw new Error(result.message || '多模态功能未启用')
          if (!result.analysis) throw new Error('图片分析未返回可用内容')
          evidenceBlocks.push(
            `【图片证据补充（视觉模型识别，需与原图核对）】\n文件：${file.name}\n原图 SHA-256：${result.image_sha256 || ''}\n${result.analysis}`,
          )
        } else {
          throw new Error(`${file.name} 暂不支持，仅支持图片、PDF、DOCX 和 TXT`)
        }
        setAttachments((current) =>
          current.map((candidate) => candidate.id === item.id ? { ...candidate, status: 'done' } : candidate),
        )
      }
      setUploading(false)
      const combinedMessage = [trimmed, ...evidenceBlocks].filter(Boolean).join('\n\n')
      let reply = ''
      await streamChat(
        { user_id: userId, session_id: sessionId, message: combinedMessage },
        (event) => {
          if (event.type === 'token') {
            reply += event.content || ''
            setMessages((current) =>
              current.map((candidate) =>
                candidate.id === assistantMessageId ? { ...candidate, content: reply } : candidate,
              ),
            )
          }
          if (event.type === 'done') {
            if (event.session_id && event.session_id !== sessionId) setSessionId(event.session_id)
            setDebug(event.debug ?? null)
            setStatistics(event.statistics ?? null)
            setDocument(event.document ?? null)
            if (event.document) {
              const sourceTemplateId = event.document.source?.template_id
              if (typeof sourceTemplateId === 'string') setSelectedTemplate(sourceTemplateId)
              setActiveTab('documents')
              setInspectorOpen(true)
            } else if (event.statistics) {
              setActiveTab('sources')
              setInspectorOpen(true)
            }
          }
          if (event.type === 'error') throw new Error(event.message || '后端返回错误')
        },
      )
      if (!reply) {
        setMessages((current) =>
          current.map((candidate) =>
            candidate.id === assistantMessageId ? { ...candidate, content: '本轮没有收到可显示的回复，请稍后重试。' } : candidate,
          ),
        )
      }
      setAttachments([])
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : '请求失败，请稍后重试'
      setError(message)
      setAttachments((current) =>
        current.map((item) =>
          files.some((queued) => queued.id === item.id)
            ? { ...item, status: 'error', note: message }
            : item,
        ),
      )
      setMessages((current) =>
        current.map((candidate) =>
          candidate.id === assistantMessageId
            ? { ...candidate, content: `连接未完成：${message}` }
            : candidate,
        ),
      )
    } finally {
      setUploading(false)
      setBusy(false)
    }
  }

  const submitDraft = () => void send(draft)

  const submitIntake = () => {
    const intakeMessage = buildIntakeMessage()
    if (!intakeMessage) {
      setError('请至少填写一项案件信息，或直接在下方输入框描述情况。')
      focusComposer()
      return
    }
    setIntake(INITIAL_INTAKE)
    void send(intakeMessage)
  }

  const currentConversationSnapshot = (): ConversationRecord => ({
    mode,
    sessionId,
    title: deriveConversationTitle(messages, mode),
    updatedAt: messages.at(-1)?.createdAt
      || conversations.find((record) => record.sessionId === sessionId)?.updatedAt
      || Date.now(),
    messages,
    draft,
    intake,
    debug,
    statistics,
    document,
    activeTab,
  })

  const updateDraft = (value: string) => {
    setDraft(value)
    setConversations((current) => {
      const existing = current.find((record) => record.sessionId === sessionId)
      const next: ConversationRecord = {
        ...currentConversationSnapshot(),
        draft: value,
        updatedAt: existing?.updatedAt || Date.now(),
      }
      const updated = [
        next,
        ...current.filter((record) => record.sessionId !== sessionId),
      ]
      saveConversations(updated)
      return updated
    })
  }

  const loadConversation = (record: ConversationRecord) => {
    setMode(record.mode)
    setSessionId(record.sessionId)
    setMessages(record.messages)
    setDraft(record.draft || '')
    setIntake(record.intake || INITIAL_INTAKE)
    setDebug(record.debug || null)
    setStatistics(record.statistics || null)
    setDocument(record.document || null)
    setActiveTab(record.activeTab || 'case')
    setAttachments([])
    setUploading(false)
    setError('')
    setInspectorOpen(false)
    const sourceTemplateId = record.document?.source?.template_id
    if (typeof sourceTemplateId === 'string') setSelectedTemplate(sourceTemplateId)
    window.requestAnimationFrame(focusComposer)
  }

  const startNewConversation = (targetMode = mode) => {
    if (busy) return
    const current = currentConversationSnapshot()
    const next = createConversation(targetMode)
    setConversations((items) => [
      next,
      current,
      ...items.filter(
        (record) =>
          record.sessionId !== current.sessionId
          && record.sessionId !== next.sessionId,
      ),
    ])
    loadConversation(next)
  }

  const switchMode = (nextMode: WorkspaceMode) => {
    if (busy || nextMode === mode) return
    const current = currentConversationSnapshot()
    const available = [
      current,
      ...conversations.filter((record) => record.sessionId !== current.sessionId),
    ]
    const target = available
      .filter((record) => record.mode === nextMode)
      .sort((left, right) => right.updatedAt - left.updatedAt)[0]
      || createConversation(nextMode, readModeSession(nextMode))
    setConversations([
      target,
      ...available.filter((record) => record.sessionId !== target.sessionId),
    ])
    loadConversation(target)
  }

  const removeConversation = async (record: ConversationRecord) => {
    if (busy) return
    const confirmed = window.confirm(
      `确定删除“${record.title}”吗？案件状态、对话检查点和本地历史将一并清除，无法恢复。`,
    )
    if (!confirmed) return
    setBusy(true)
    setError('')
    try {
      const result = await deleteConversation(userId, record.sessionId)
      const remaining = conversations.filter(
        (item) => item.sessionId !== record.sessionId,
      )
      setConversations(remaining)
      if (record.sessionId === sessionId) {
        const next = remaining
          .filter((item) => item.mode === mode)
          .sort((left, right) => right.updatedAt - left.updatedAt)[0]
          || createConversation(mode)
        setConversations((items) => [
          next,
          ...items.filter((item) => item.sessionId !== next.sessionId),
        ])
        loadConversation(next)
      }
      if (result.warnings?.length) {
        setError(`对话已删除，但${result.warnings.join('、')}。`)
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '删除对话失败')
    } finally {
      setBusy(false)
    }
  }

  const setScenario = (text: string) => {
    updateDraft(text)
    focusComposer()
  }

  const scenarios = mode === 'case' ? CASE_SCENARIOS : QA_SCENARIOS

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-symbol"><Scale size={21} /></div>
          <div>
            <strong>法护通</strong>
            <span>法律咨询与维权辅助</span>
          </div>
        </div>
        <div className="topbar-center">
          <div className="mode-switch" role="tablist" aria-label="工作模式">
            <button
              role="tab"
              aria-selected={mode === 'qa'}
              className={mode === 'qa' ? 'active' : ''}
              onClick={() => switchMode('qa')}
              disabled={busy}
            >
              <BookOpen size={14} />
              法律问答
            </button>
            <button
              role="tab"
              aria-selected={mode === 'case'}
              className={mode === 'case' ? 'active' : ''}
              onClick={() => switchMode('case')}
              disabled={busy}
            >
              <HeartHandshake size={14} />
              案件维权
            </button>
          </div>
          <span className="topbar-divider" />
          <span className="topbar-hint">内容仅供参考，重要决定前请核验来源</span>
        </div>
        <div className="topbar-actions">
          <button className="secondary-button inspector-toggle" onClick={() => setInspectorOpen(true)}>
            <PanelRight size={16} />
            {mode === 'case' ? '案件资料' : '问答资料'}
          </button>
          <button className="secondary-button" onClick={() => startNewConversation()} disabled={busy}>
            <MessageSquarePlus size={16} />
            {mode === 'case' ? '新建案件' : '新建问答'}
          </button>
          <button className="avatar-button" title="当前为匿名工作区"><span>访</span></button>
        </div>
      </header>

      <main className="workspace-grid">
        <aside className="sidebar">
          <section className="case-card">
            <div className="case-card-top">
              <div className="case-badge">
                {mode === 'case' ? <HeartHandshake size={17} /> : <BookOpen size={17} />}
              </div>
              <span className="case-open">{mode === 'case' ? 'CASE' : 'Q&A'}</span>
            </div>
            <p className="section-kicker">CURRENT WORKSPACE</p>
            <h1>
              {debug?.domain
                ? DOMAIN_LABELS[debug.domain] || debug.domain
                : mode === 'case' ? '未命名维权案件' : '法律知识问答'}
            </h1>
            <p className="case-id-note">
              {mode === 'case'
                ? '案件长期保留，切换或刷新后可继续。'
                : '问答独立保存，不会写入维权案件。'}
            </p>
            <div className="case-card-footer">
              {mode === 'case'
                ? <ConfidenceBadge tier={debug?.confidence_tier} />
                : <span className="confidence-badge qa">知识检索</span>}
              <span className="round-note">{messages.filter((item) => item.role === 'user').length} 轮输入</span>
            </div>
          </section>

          <ConversationManager
            mode={mode}
            conversations={conversations}
            activeSessionId={sessionId}
            onOpen={loadConversation}
            onCreate={() => startNewConversation()}
            onDelete={(record) => void removeConversation(record)}
          />

          <section className="side-section quick-section">
            <div className="side-section-head">
              <div>
                <p className="section-kicker">QUICK START</p>
                <h2>常见入口</h2>
              </div>
              <Sparkles size={16} className="muted-icon" />
            </div>
            <div className="scenario-list">
              {scenarios.map((scenario) => (
                <button key={scenario.label} className="scenario-row" onClick={() => setScenario(scenario.text)}>
                  <span>{scenario.label}</span>
                  <ArrowUp size={14} />
                </button>
              ))}
            </div>
          </section>

          <HealthCard health={health} loading={healthLoading} onRefresh={() => void refreshHealth()} />

          <section className="side-section privacy-card">
            <div className="privacy-icon"><ShieldCheck size={16} /></div>
            <div>
              <strong>隐私提示</strong>
              <p>
                {mode === 'case'
                  ? '案件长期保留至手动删除；上传前请遮挡无关的敏感信息。'
                  : '问答记录仅在当前匿名浏览器中管理，重要信息请谨慎输入。'}
              </p>
            </div>
          </section>
        </aside>

        <section className="conversation-column">
          {mode === 'case' ? (
            <IntakePanel intake={intake} setIntake={setIntake} onSubmit={submitIntake} busy={busy} />
          ) : null}
          <div className="conversation-header">
            <div>
              <p className="section-kicker">CONVERSATION</p>
              <h2>{mode === 'case' ? '案件对话' : '法律问答'}</h2>
            </div>
            {busy ? <span className="processing-label"><LoaderCircle size={14} className="spin" />正在处理</span> : null}
          </div>
          <div className="conversation-scroll">
            {messages.length === 0 ? (
              <WelcomeState mode={mode} scenarios={scenarios} onScenario={setScenario} />
            ) : (
              <div className="message-list">
                {messages.map((message) => <MessageCard key={message.id} message={message} />)}
                <div ref={conversationEndRef} aria-hidden="true" />
              </div>
            )}
          </div>
          <div className="composer-area">
            {error ? (
              <div className="error-banner">
                <AlertTriangle size={16} />
                <span>{error}</span>
                <button className="icon-button subtle" title="关闭提示" onClick={() => setError('')}><X size={15} /></button>
              </div>
            ) : null}
            {attachments.length ? (
              <div className="attachment-strip">
                {attachments.map((item) => (
                  <div className={`pending-file ${item.status}`} key={item.id}>
                    {item.file.type.startsWith('image/') ? <ImagePlus size={14} /> : <FileText size={14} />}
                    <span title={item.file.name}>{item.file.name}</span>
                    <small>{formatBytes(item.file.size)}</small>
                    {item.status === 'processing' ? <LoaderCircle size={13} className="spin" /> : null}
                    {item.status === 'done' ? <Check size={13} /> : null}
                    <button className="remove-file" title={`移除 ${item.file.name}`} onClick={() => removeAttachment(item.id)} disabled={busy}><X size={13} /></button>
                  </div>
                ))}
              </div>
            ) : null}
            <div className="composer">
              <input
                ref={fileInputRef}
                className="sr-only"
                type="file"
                multiple
                accept="image/*,.pdf,.docx,.txt"
                onChange={(event) => handleFileSelection(event.target.files)}
              />
              <button className="composer-tool" title="添加图片、PDF、DOCX 或 TXT" onClick={() => fileInputRef.current?.click()} disabled={busy}>
                <Paperclip size={19} />
              </button>
              <textarea
                ref={textAreaRef}
                value={draft}
                onChange={(event) => updateDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    submitDraft()
                  }
                }}
                placeholder={
                  mode === 'case'
                    ? '描述案件进展、补充事实或上传证据……'
                    : '询问法律规定、办理流程、类案或统计数据……'
                }
                rows={1}
                disabled={busy}
              />
              <button className="send-button" title="发送消息" onClick={submitDraft} disabled={busy || (!draft.trim() && !attachments.length)}>
                {uploading ? <LoaderCircle size={18} className="spin" /> : <ArrowUp size={19} />}
              </button>
            </div>
            <div className="composer-footer">
              <span>Enter 发送 · Shift + Enter 换行</span>
              <span><LockIcon /> 当前为匿名会话</span>
            </div>
          </div>
          {mode === 'case' ? (
            <div className="action-rail">
              <button onClick={() => void send('现在生成方案')} disabled={busy}><Check size={15} />现在生成方案</button>
              <button onClick={() => void send('继续补充')} disabled={busy}><RefreshCw size={15} />继续补充</button>
              <button onClick={() => void send('生成文书')} disabled={busy}><FileText size={15} />生成参考文书</button>
            </div>
          ) : null}
        </section>

        <button
          className={`inspector-backdrop ${inspectorOpen ? 'open' : ''}`}
          aria-label="关闭侧栏遮罩"
          onClick={() => setInspectorOpen(false)}
        />
        <Inspector
          mode={mode}
          open={inspectorOpen}
          onClose={() => setInspectorOpen(false)}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          debug={debug}
          statistics={statistics}
          document={document}
          templates={templates}
          selectedTemplate={selectedTemplate}
          setSelectedTemplate={setSelectedTemplate}
          templateError={templateError}
        />
      </main>
      <footer className="app-footer">
        <span>
          法护通 · {mode === 'case' ? '案件状态长期保留至手动删除' : '法律问答与维权案件相互隔离'}
        </span>
        <span>法律信息与行动参考，不替代律师或办案机关的正式判断</span>
      </footer>
    </div>
  )
}

function LockIcon() {
  return <ShieldCheck size={13} />
}

export default App
