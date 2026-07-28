"""
从中国裁判文书网爬取真实民事案例

使用策略：
1. 使用 requests + BeautifulSoup 解析页面
2. 针对每个领域使用特定关键词搜索
3. 提取案件标题、案号、案情、裁判要旨、法院、日期等字段
4. 每个领域爬取 150-200 条

技术要点：
- 裁判文书网有反爬机制，需要设置合理的请求头和延迟
- 使用 Selenium 模拟浏览器访问（如果 API 不可用）
- 数据清洗：去除 HTML 标签、统一编码
"""
import asyncio
import time
import random
import sys
from pathlib import Path
from typing import List, Dict
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

from src.infra.database import AsyncSessionLocal
from src.modules.legal.model import LegalCase


# 搜索关键词配置
# domain 必须使用 prompts.py 的规范域码，否则 case_rag 按 domain 过滤时召回不到
SEARCH_CONFIG = {
    "real_estate": {
        "keywords": ["房屋租赁合同纠纷", "房屋买卖合同纠纷", "物业服务合同纠纷"],
        "domain": "contracts_property_housing",
        "target": 150
    },
    "labor_social_security": {
        "keywords": ["劳动争议", "劳动合同纠纷", "追索劳动报酬"],
        "domain": "labor_social_security",
        "target": 150
    },
    "consumer_market": {
        "keywords": ["买卖合同纠纷", "产品责任纠纷", "网络购物合同纠纷"],
        "domain": "consumer_market",
        "target": 150
    },
    "family_marriage": {
        "keywords": ["离婚纠纷", "抚养费纠纷", "继承纠纷"],
        "domain": "family_vulnerable_groups",
        "target": 100
    },
    "contract_commercial": {
        "keywords": ["合同纠纷", "借款合同纠纷", "买卖合同纠纷"],
        "domain": "contracts_property_housing",
        "target": 100
    }
}


class WenshuCrawler:
    """裁判文书网爬虫"""

    def __init__(self):
        self.base_url = "https://wenshu.court.gov.cn"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })

    def init_selenium(self):
        """初始化 Selenium WebDriver"""
        options = Options()
        options.add_argument("--headless")  # 无头模式
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument(f"user-agent={self.session.headers['User-Agent']}")

        try:
            self.driver = webdriver.Chrome(options=options)
            print("[INFO] Selenium WebDriver initialized")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize Selenium: {e}")
            print("[INFO] Please install ChromeDriver: https://chromedriver.chromium.org/")
            return False

    def search_cases(self, keyword: str, limit: int = 50) -> List[Dict]:
        """搜索案例（使用 Selenium）"""
        if not hasattr(self, 'driver'):
            if not self.init_selenium():
                return []

        cases = []
        try:
            # 访问搜索页面
            search_url = f"{self.base_url}/website/wenshu/181217BMTKHNT2W0/index.html"
            self.driver.get(search_url)
            time.sleep(3)  # 等待页面加载

            # 输入关键词搜索
            search_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "searchWord"))
            )
            search_input.clear()
            search_input.send_keys(keyword)

            # 点击搜索按钮
            search_btn = self.driver.find_element(By.CLASS_NAME, "search-button")
            search_btn.click()
            time.sleep(5)  # 等待搜索结果

            # 解析搜索结果
            page = 1
            while len(cases) < limit and page <= 5:  # 最多爬取 5 页
                print(f"  Crawling page {page} for '{keyword}'...")

                # 获取当前页面的案例列表
                case_items = self.driver.find_elements(By.CLASS_NAME, "LM_list")

                for item in case_items:
                    if len(cases) >= limit:
                        break

                    try:
                        # 提取案例信息
                        title_elem = item.find_element(By.CLASS_NAME, "caseName")
                        title = title_elem.text.strip()

                        # 点击进入详情页
                        title_elem.click()
                        time.sleep(2)

                        # 切换到新窗口
                        self.driver.switch_to.window(self.driver.window_handles[-1])

                        # 提取详细信息
                        case_data = self._parse_case_detail()
                        if case_data:
                            case_data["title"] = title
                            cases.append(case_data)
                            print(f"    [+] Extracted: {title[:50]}...")

                        # 关闭详情页，返回列表
                        self.driver.close()
                        self.driver.switch_to.window(self.driver.window_handles[0])
                        time.sleep(random.uniform(1, 2))

                    except Exception as e:
                        print(f"    [!] Failed to extract case: {e}")
                        continue

                # 翻页
                try:
                    next_btn = self.driver.find_element(By.CLASS_NAME, "nextpage")
                    next_btn.click()
                    time.sleep(3)
                    page += 1
                except:
                    print(f"  No more pages or pagination failed")
                    break

        except Exception as e:
            print(f"[ERROR] Search failed for '{keyword}': {e}")

        return cases

    def _parse_case_detail(self) -> Dict | None:
        """解析案例详情页"""
        try:
            # 等待页面加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "PDF-view"))
            )

            # 提取案号
            case_number = ""
            try:
                case_number_elem = self.driver.find_element(By.XPATH, "//div[contains(text(), '案号')]")
                case_number = case_number_elem.text.replace("案号：", "").strip()
            except:
                pass

            # 提取法院
            court = ""
            try:
                court_elem = self.driver.find_element(By.XPATH, "//div[contains(text(), '审理法院')]")
                court = court_elem.text.replace("审理法院：", "").strip()
            except:
                pass

            # 提取案由
            cause = ""
            try:
                cause_elem = self.driver.find_element(By.XPATH, "//div[contains(text(), '案由')]")
                cause = cause_elem.text.replace("案由：", "").strip()
            except:
                pass

            # 提取判决日期
            judge_date = ""
            try:
                date_elem = self.driver.find_element(By.XPATH, "//div[contains(text(), '裁判日期')]")
                judge_date = date_elem.text.replace("裁判日期：", "").strip()
            except:
                pass

            # 提取案情（从 PDF 查看器中提取文本）
            facts = ""
            gist = ""
            try:
                pdf_text = self.driver.find_element(By.CLASS_NAME, "PDF-view").text
                # 简单处理：前 500 字作为案情，后续优化可以用 NLP 提取
                facts = pdf_text[:500] if len(pdf_text) > 500 else pdf_text
                # 尝试提取裁判要旨（通常在结尾部分）
                if "本院认为" in pdf_text:
                    gist_start = pdf_text.index("本院认为")
                    gist = pdf_text[gist_start:gist_start+300]
            except:
                pass

            if not facts:
                return None

            return {
                "case_number": case_number,
                "court": court,
                "cause": cause,
                "judge_date": judge_date,
                "facts": facts,
                "gist": gist
            }

        except Exception as e:
            print(f"      [!] Parse detail failed: {e}")
            return None

    def close(self):
        """关闭浏览器"""
        if hasattr(self, 'driver'):
            self.driver.quit()


async def import_cases_to_db(cases: List[Dict], domain: str):
    """将爬取的案例导入数据库"""
    async with AsyncSessionLocal() as session:
        added = 0
        for case_data in cases:
            case = LegalCase(
                title=case_data.get("title", ""),
                cause=case_data.get("cause", ""),
                facts=case_data["facts"],
                gist=case_data.get("gist", ""),
                court=case_data.get("court", ""),
                domain=domain,
                source="wenshu_court"
            )
            session.add(case)
            added += 1

        await session.commit()
        print(f"[SUCCESS] Imported {added} cases for domain '{domain}'")


async def main():
    print("=" * 60)
    print("Crawl Civil Cases from wenshu.court.gov.cn")
    print("=" * 60)
    print()

    crawler = WenshuCrawler()

    try:
        for domain_key, config in SEARCH_CONFIG.items():
            print(f"\n[DOMAIN] {config['domain']}")
            print(f"[TARGET] {config['target']} cases")

            all_cases = []
            for keyword in config["keywords"]:
                print(f"\n[KEYWORD] {keyword}")
                cases = crawler.search_cases(keyword, limit=config["target"] // len(config["keywords"]) + 20)
                all_cases.extend(cases)

                if len(all_cases) >= config["target"]:
                    break

            # 去重（根据标题）
            unique_cases = []
            seen_titles = set()
            for case in all_cases:
                if case.get("title") not in seen_titles:
                    unique_cases.append(case)
                    seen_titles.add(case.get("title"))

            # 截取目标数量
            unique_cases = unique_cases[:config["target"]]

            print(f"\n[RESULT] Extracted {len(unique_cases)} unique cases")

            # 导入数据库
            if unique_cases:
                await import_cases_to_db(unique_cases, config["domain"])

            # 保存到本地（备份）
            backup_file = ROOT / f"data/cases_{config['domain']}.json"
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(unique_cases, f, ensure_ascii=False, indent=2)
            print(f"[BACKUP] Saved to {backup_file}")

    finally:
        crawler.close()

    print("\n" + "=" * 60)
    print("Crawling Completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
