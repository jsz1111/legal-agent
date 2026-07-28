import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.build_case_pilot_dataset import (
    build_retrieval_text,
    clean_text,
    extract_sections,
    screen_row,
    write_sqlite,
)


def _row(text: str) -> dict[str, str]:
    return {
        "原始链接": "https://wenshu.court.gov.cn/example",
        "案号": "（2021）测民初1号",
        "案件名称": "甲与乙劳动争议民事判决书",
        "法院": "测试人民法院",
        "所属地区": "测试省",
        "案件类型": "民事案件",
        "案件类型编码": "1",
        "来源": "www.example.invalid",
        "审理程序": "民事一审",
        "裁判日期": "2021-10-01",
        "公开日期": "2021-10-02",
        "当事人": "甲；乙",
        "案由": "劳动争议",
        "法律依据": "劳动合同法第三十条",
        "全文": text,
    }


class CasePilotDatasetTests(unittest.TestCase):
    def test_clean_text_removes_vendor_watermark(self):
        value = "文书内容本 院 认 为，应当支付。微信公众号“马克 数据网”"
        cleaned = clean_text(value)
        self.assertEqual(cleaned, "本院认为，应当支付。")

    def test_clean_text_removes_spaced_and_html_vendor_watermark(self):
        value = "判决生效。审 判 员&#xa0;张三 来源：百度“马 克 数 据 网”"
        cleaned = clean_text(value)
        self.assertNotIn("马克数据网", cleaned)
        self.assertNotIn("来源：百度", cleaned)
        self.assertNotIn("&#xa0;", cleaned)

    def test_clean_text_removes_vendor_domain_footers(self):
        variants = (
            "判决生效。来自：www.macrodatas.cn",
            "判决生效。更多数据：www.macrodatas.cn",
            "判决生效。更多数据：搜索来源：www.macrodatas.cn",
            "判决生效。更多数据：搜索“马克数据网”来源：www.macrodatas.cn",
            "判决生效。 - 来源：https://www.macrodatas.cn/detail",
        )
        for value in variants:
            with self.subTest(value=value):
                self.assertEqual(clean_text(value), "判决生效。")

    def test_extract_sections_and_retrieval_text(self):
        text = (
            "诉讼请求：请求支付拖欠工资。" + "双方争议事实。" * 30
            + "本院认为" + "现有证据能够证明劳动关系。" * 20
            + "判决如下" + "被告支付拖欠工资。" * 10
            + "审判员张三书记员李四"
        )
        sections = extract_sections(text)
        self.assertIsNotNone(sections)
        retrieval = build_retrieval_text(_row(text), sections)
        self.assertIn("案由：劳动争议", retrieval)
        self.assertIn("法院认为：", retrieval)
        self.assertIn("裁判结果：", retrieval)
        self.assertLessEqual(len(retrieval), 800)

    def test_screen_row_removes_source_and_preserves_url(self):
        text = (
            "诉讼请求：请求支付拖欠工资。" + "案件事实清楚。" * 100
            + "本院认为" + "现有证据能够证明劳动关系。" * 30
            + "判决如下" + "被告支付拖欠工资。" * 10
            + "审判员张三"
        )
        row, reason = screen_row(_row(text), min_length=500, max_length=8000)
        self.assertEqual(reason, "合格")
        self.assertIsNotNone(row)
        self.assertNotIn("来源", row)
        self.assertEqual(row["原始链接"], "https://wenshu.court.gov.cn/example")

    def test_procedural_result_is_rejected(self):
        text = (
            "诉讼请求：请求处理纠纷。" + "案件事实。" * 100
            + "本院认为" + "符合法律规定。" * 30
            + "判决如下准许撤回起诉，本案案件受理费由原告负担，其他费用不再收取。"
        )
        row, reason = screen_row(_row(text), min_length=500, max_length=8000)
        self.assertIsNone(row)
        self.assertEqual(reason, "程序性裁判")

    def test_sqlite_preserves_metadata_without_source(self):
        text = (
            "诉讼请求：请求支付拖欠工资。" + "案件事实清楚。" * 100
            + "本院认为" + "现有证据能够证明劳动关系。" * 30
            + "判决如下" + "被告支付拖欠工资。" * 10
            + "审判员张三"
        )
        row, _ = screen_row(_row(text), min_length=500, max_length=8000)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "cases.sqlite3"
            write_sqlite(database, [row])
            connection = sqlite3.connect(database)
            try:
                columns = {
                    item[1]
                    for item in connection.execute("PRAGMA table_info(legal_cases)")
                }
                record = connection.execute(
                    "SELECT original_url, cause, full_text, retrieval_text FROM legal_cases"
                ).fetchone()
            finally:
                connection.close()
        self.assertNotIn("source", columns)
        self.assertEqual(record[0], "https://wenshu.court.gov.cn/example")
        self.assertEqual(record[1], "劳动争议")
        self.assertTrue(record[2])
        self.assertTrue(record[3])


if __name__ == "__main__":
    unittest.main()
