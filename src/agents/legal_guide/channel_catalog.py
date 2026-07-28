"""维权渠道的地区标准化、分层排序和降级数据。"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class RegionInfo:
    code: str
    name: str
    legacy_codes: tuple[str, ...] = ()


_REGIONS = {
    "110000": RegionInfo("110000", "北京", ("BJ",)),
    "310000": RegionInfo("310000", "上海", ("SH",)),
}

_REGION_ALIASES = {
    "北京": "110000",
    "北京市": "110000",
    "bj": "110000",
    "上海": "310000",
    "上海市": "310000",
    "sh": "310000",
}

_DOMAIN_ALIASES = {
    "contract_commercial": "contracts_property_housing",
    "real_estate_construction": "contracts_property_housing",
    "family_marriage": "family_vulnerable_groups",
    "traffic_accident": "traffic_personal_injury",
    "medical_dispute": "medical_education_tax",
    "administrative": "administrative_remedies",
}

_DOMAIN_LABELS = {
    "labor_social_security": "劳动和社会保障争议",
    "consumer_market": "消费维权",
    "contracts_property_housing": "合同、房产和租赁争议",
    "criminal_public_security": "刑事和治安问题",
    "family_vulnerable_groups": "婚姻家庭和弱势群体保护",
    "traffic_personal_injury": "交通事故和人身损害",
    "medical_education_tax": "医疗、教育和税务争议",
    "administrative_remedies": "行政救济",
    "intellectual_property": "知识产权",
    "environment_pollution": "环境污染",
    "cyber_data_fraud": "网络、数据和诈骗问题",
    "mediation_notary_arbitration": "调解、公证和仲裁",
}

_GENERAL_DOMAINS = (
    "public_legal_service",
    "general_government_service",
)


def resolve_region(raw: str | None) -> RegionInfo:
    """当前试点只精确识别北京、上海，其余地区安全降级为全国。"""
    value = str(raw or "").strip()
    if not value or value.upper() == "CN":
        return RegionInfo("CN", "全国")
    lowered = value.lower()
    code = _REGION_ALIASES.get(lowered) or _REGION_ALIASES.get(value)
    if code:
        return _REGIONS[code]
    if value in _REGIONS:
        return _REGIONS[value]
    for token, region_code in (("北京", "110000"), ("上海", "310000")):
        if token in value:
            return _REGIONS[region_code]
    return RegionInfo("CN", "全国")


def extract_supported_region(text: str) -> str:
    """仅从明确行政区名称中提取试点地区，禁止自由文本被误判为地名。"""
    value = str(text or "")
    if "北京" in value:
        return "北京"
    if "上海" in value:
        return "上海"
    return ""


def normalize_region_name(raw: str | None) -> str:
    """返回状态机可保存的地区名；非试点地区返回空字符串。"""
    info = resolve_region(raw)
    return "" if info.code == "CN" else info.name


def normalize_domain(domain: str | None) -> str:
    value = str(domain or "").strip()
    return _DOMAIN_ALIASES.get(value, value)


def channel_query_domains(domain: str | None) -> tuple[str, ...]:
    normalized = normalize_domain(domain)
    values = ([normalized] if normalized and normalized != "other" else []) + list(_GENERAL_DOMAINS)
    return tuple(dict.fromkeys(values))


def channel_query_region_codes(region: str | None) -> tuple[str, ...]:
    info = resolve_region(region)
    return tuple(dict.fromkeys((info.code, *info.legacy_codes, "CN")))


def _contact_key(channel: dict) -> tuple[str, str]:
    """热线按号码去重；无电话的官网按主机和路径去重。"""
    phone = "".join(ch for ch in str(channel.get("phone") or "") if ch.isdigit())
    if phone:
        return "phone", phone
    url = str(channel.get("url") or channel.get("source_url") or "").strip()
    if url:
        parsed = urlsplit(url)
        return "url", f"{parsed.netloc.lower()}{parsed.path.rstrip('/').lower()}"
    return "name", str(channel.get("name") or "").strip().lower()


def _stage(domain: str) -> tuple[int, str]:
    if domain in {"public_legal_service", "general"}:
        return 2, "法律咨询兜底"
    if domain == "general_government_service":
        return 3, "协调转办"
    return 1, "优先办理"


def _reason(channel: dict, requested_domain: str, region: RegionInfo) -> str:
    domain = str(channel.get("domain") or "")
    is_local = region.code != "CN" and str(channel.get("region_code") or "") in {
        region.code,
        *region.legacy_codes,
    }
    if domain in {"public_legal_service", "general"}:
        return f"{region.name if is_local else '全国'}公共法律服务，可用于法律咨询和法律援助转介"
    if domain == "general_government_service":
        return "专属渠道未解决时，可由政务服务热线协调或转交主管部门"
    label = _DOMAIN_LABELS.get(requested_domain, "当前争议")
    return f"适用于{label}" + (f"，且属于{region.name}本地办理入口" if is_local else "的全国通用入口")


def rank_channel_candidates(
    channels: list[dict],
    domain: str | None,
    region: str | None,
    limit: int = 6,
) -> list[dict]:
    """按专属、本地公共法律服务、政务兜底分层，并对重复热线去重。"""
    requested_domain = normalize_domain(domain)
    region_info = resolve_region(region)

    def sort_key(channel: dict) -> tuple[int, int, int, str]:
        channel_domain = str(channel.get("domain") or "")
        stage_order, _ = _stage(channel_domain)
        code = str(channel.get("region_code") or "")
        local_order = 0 if region_info.code != "CN" and code in {
            region_info.code,
            *region_info.legacy_codes,
        } else 1
        return stage_order, local_order, int(channel.get("priority") or 100), str(channel.get("name") or "")

    selected: list[dict] = []
    seen: set[tuple[str, str]] = set()
    stage_counts = {1: 0, 2: 0, 3: 0}
    # 防止某一层的多个网页占满结果，至少为法律咨询和协调转办保留位置。
    stage_caps = {1: 3, 2: 2, 3: 1}
    for raw in sorted(channels, key=sort_key):
        if str(raw.get("status") or "active") != "active":
            continue
        stage_order, stage_label = _stage(str(raw.get("domain") or ""))
        if stage_counts[stage_order] >= stage_caps[stage_order]:
            continue
        key = _contact_key(raw)
        if key in seen:
            continue
        seen.add(key)
        item = dict(raw)
        item["route_stage"] = stage_label
        item["recommendation_reason"] = _reason(item, requested_domain, region_info)
        item["resolved_region"] = region_info.name
        selected.append(item)
        stage_counts[stage_order] += 1
        if len(selected) >= limit:
            break
    return selected


def fallback_channels(domain: str | None, region: str | None, limit: int = 6) -> list[dict]:
    """数据库不可用时的最小可靠兜底，内容与试点数据包保持一致。"""
    normalized = normalize_domain(domain)
    rows: list[dict] = []
    if normalized == "consumer_market":
        rows.append({
            "name": "全国12315平台", "domain": normalized, "phone": "12315",
            "url": "https://www.12315.cn/", "region_code": "CN", "priority": 10,
            "applicable_matters": ["消费投诉", "市场监管违法线索"],
            "required_materials": ["订单或付款凭证", "商品或服务问题证据", "与经营者协商记录"],
            "source_org": "国家市场监督管理总局", "source_url": "https://www.12315.cn/",
        })
    elif normalized == "labor_social_security":
        rows.append({
            "name": "全国人社政务服务平台", "domain": normalized, "phone": "12333",
            "url": "https://12333.gov.cn/", "region_code": "CN", "priority": 10,
            "applicable_matters": ["劳动保障政策咨询", "劳动争议服务查询"],
            "required_materials": ["身份证明", "劳动关系证明", "工资或考勤材料"],
            "source_org": "人力资源和社会保障部", "source_url": "https://12333.gov.cn/",
        })
    rows.extend([
        {
            "name": "12348中国法律服务网", "domain": "public_legal_service", "phone": "12348",
            "url": "https://www.12348.gov.cn/", "region_code": "CN", "priority": 30,
            "applicable_matters": ["公共法律咨询", "法律援助指引"],
            "required_materials": ["案情时间线", "已有证据清单", "希望解决的诉求"],
            "source_org": "司法部", "source_url": "https://www.12348.gov.cn/",
        },
        {
            "name": "12345政务服务便民热线", "domain": "general_government_service", "phone": "12345",
            "url": "https://tousu.www.gov.cn/zwfw/index.htm", "region_code": "CN", "priority": 40,
            "applicable_matters": ["主管部门不明确时的咨询转办", "政务服务投诉与建议"],
            "required_materials": ["事项经过", "已联系部门和受理编号", "明确诉求"],
            "source_org": "国务院办公厅", "source_url": "https://tousu.www.gov.cn/zwfw/index.htm",
        },
    ])
    return rank_channel_candidates(rows, normalized, region, limit=limit)
