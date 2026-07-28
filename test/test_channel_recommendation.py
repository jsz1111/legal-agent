"""维权渠道地区标准化、分层排序和输出细节测试。"""
from src.agents.legal_guide.channel_catalog import (
    extract_supported_region,
    rank_channel_candidates,
    resolve_region,
)
from src.agents.legal_guide.formatters import fmt_channels


def test_pilot_region_normalization_is_strict():
    assert resolve_region("北京市朝阳区").code == "110000"
    assert resolve_region("SH").code == "310000"
    assert resolve_region("在饭馆吃东西").code == "CN"
    assert extract_supported_region("我在北京朝阳区上班") == "北京"
    assert extract_supported_region("我在饭馆吃东西发现异物") == ""


def test_channels_are_layered_local_first_and_deduplicated():
    rows = [
        {
            "name": "重复12315说明文件", "domain": "consumer_market", "phone": "12315",
            "region_code": "CN", "priority": 100, "status": "active",
        },
        {
            "name": "全国12315平台", "domain": "consumer_market", "phone": "12315",
            "region_code": "CN", "priority": 10, "status": "active",
        },
        {
            "name": "北京法律服务网", "domain": "public_legal_service", "phone": "12348",
            "region_code": "110000", "priority": 20, "status": "active",
        },
        {
            "name": "全国法律服务网", "domain": "public_legal_service", "phone": "12348",
            "region_code": "CN", "priority": 30, "status": "active",
        },
        {
            "name": "北京12345", "domain": "general_government_service", "phone": "12345",
            "region_code": "110000", "priority": 40, "status": "active",
        },
    ]

    result = rank_channel_candidates(rows, "consumer_market", "北京")

    assert [item["name"] for item in result] == [
        "全国12315平台",
        "北京法律服务网",
        "北京12345",
    ]
    assert [item["route_stage"] for item in result] == [
        "优先办理",
        "法律咨询兜底",
        "协调转办",
    ]
    assert "北京" in result[1]["recommendation_reason"]


def test_channel_formatter_includes_action_details_and_source():
    output = fmt_channels([{
        "name": "全国12315平台",
        "route_stage": "优先办理",
        "recommendation_reason": "适用于消费维权",
        "phone": "12315",
        "url": "https://www.12315.cn/",
        "applicable_matters": ["消费投诉"],
        "required_materials": ["订单", "协商记录"],
        "service_hours": "以平台公示为准",
        "source_org": "国家市场监督管理总局",
        "source_url": "https://www.12315.cn/",
        "last_verified_on": "2026-07-27",
    }])

    assert "推荐理由" in output
    assert "先准备：订单；协商记录" in output
    assert "国家市场监督管理总局" in output
    assert "核验于2026-07-27" in output


def test_layer_caps_keep_public_legal_and_government_fallbacks():
    rows = [
        {
            "name": f"劳动入口{i}", "domain": "labor_social_security", "phone": f"1233{i}",
            "region_code": "CN", "priority": i, "status": "active",
        }
        for i in range(6)
    ] + [
        {
            "name": "12348", "domain": "public_legal_service", "phone": "12348",
            "region_code": "CN", "priority": 30, "status": "active",
        },
        {
            "name": "12345", "domain": "general_government_service", "phone": "12345",
            "region_code": "CN", "priority": 40, "status": "active",
        },
    ]

    result = rank_channel_candidates(rows, "labor_social_security", "CN")

    assert sum(item["route_stage"] == "优先办理" for item in result) == 3
    assert any(item["route_stage"] == "法律咨询兜底" for item in result)
    assert any(item["route_stage"] == "协调转办" for item in result)
