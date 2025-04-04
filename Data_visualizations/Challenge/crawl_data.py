import time
from tqdm import tqdm
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

chrome_options = Options()
chrome_options.add_argument('--headless') 
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)


def crawl_data(year, areacode):
    sbd_list = [areacode + f'{j:06}' for j in range(1, 1000)] 
    data_list = []

    for sbd in tqdm(sbd_list, desc="Đang thu thập dữ liệu", unit="sbd"):
        url = f'https://diemthi.vnanet.vn/Home/SearchBySobaodanh?code={sbd}&nam={year}'

        browser.get(url)
        time.sleep(2)

        try:
            page_source = browser.page_source
            start_index = page_source.find("{")
            end_index = page_source.rfind("}") + 1
            json_data = page_source[start_index:end_index]
            data = json.loads(json_data)

            if data.get("result"):
                data_list.append(data["result"][0])
        except Exception as e:
            print(f"failed to get data SBD {sbd}: {e}")

    df = pd.DataFrame(data_list)

    print(df)
    browser.quit()
    df.to_csv(f'./Data_visualizations/Challenge/diem_thi_{areacode}_{year}.csv', index=False)

if __name__== "__main__":
    print('Starting crawl ... !')
    crawl_data(2024, '03')