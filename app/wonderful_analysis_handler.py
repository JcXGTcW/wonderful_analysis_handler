import io
import os
import json
import logging
import platform
import requests
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats

from mcp.server.fastmcp import FastMCP, Image

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("data_analysis_mcp")
logger.info("程式啟動，日誌記錄開始")

# 建立常用的中文字體列表，按優先順序排列
COMMON_CHINESE_FONTS = [
    'Microsoft JhengHei', 'Microsoft YaHei',  # 優先使用微軟正黑體和雅黑體
    'SimSun', 'SimHei', 'NSimSun',             # 其次是宋體和黑體
    'DFKai-SB', 'KaiTi', 'FangSong',           # 再次是楷體和仿宋
    'Noto Sans CJK TC', 'Noto Sans TC',        # 再次是Google Noto字體
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',# 文泉驛字體
    'Arial Unicode MS'                          # 最後是通用字體
]

# 建立不支援中文的字體黑名單
BLACKLISTED_FONTS = [
    'Microsoft Himalaya', 
    'Noto Serif Hebrew', 
    'David', 
    'Segoe UI Historic', 
    'Leelawadee UI'
]

def get_system_font_paths() -> List[str]:
    """獲取當前系統可能的中文字體路徑"""
    system = platform.system()
    font_paths = []
    
    if system == "Windows":
        font_dir = os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'Fonts')
        font_paths = [
            os.path.join(font_dir, 'msjh.ttc'),  # 微軟正黑體
            os.path.join(font_dir, 'mingliu.ttc'),  # 細明體
            os.path.join(font_dir, 'simsun.ttc'),  # 新細明體
            os.path.join(font_dir, 'kaiu.ttf'),  # 標楷體
        ]
    elif system == "Darwin":  # macOS
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Microsoft/PMingLiU.ttf'
        ]
    elif system == "Linux":
        font_paths = [
            '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc',
            '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/arphic/uming.ttc'
        ]
    
    return [path for path in font_paths if os.path.exists(path)]

def get_available_chinese_fonts() -> List[str]:
    """取得系統已安裝的中文字體列表，依優先順序排列"""
    # 1. 先找出所有可能的中文字體
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 2. 篩選出黑名單外的中文字體
    chinese_fonts = [
        f for f in all_fonts 
        if any(name in f for name in ['Heiti', 'Hei', 'Ming', 'Song', 'Kai', 'Microsoft', 
                                      'WenQuanYi', 'Noto', 'DFKai', 'PMingLiu', 'SimSun', 
                                      'NSimSun', 'Arial Unicode MS', 'Yu Gothic', 'Meiryo']) 
        and f not in BLACKLISTED_FONTS
    ]
    
    # 3. 按照COMMON_CHINESE_FONTS的順序排列找到的字體
    sorted_fonts = []
    for preferred_font in COMMON_CHINESE_FONTS:
        exact_matches = [f for f in chinese_fonts if f == preferred_font]
        partial_matches = [f for f in chinese_fonts if preferred_font in f and f not in exact_matches]
        sorted_fonts.extend(exact_matches)
        sorted_fonts.extend(partial_matches)
    
    # 4. 添加所有未排序的中文字體到列表末尾
    remaining_fonts = [f for f in chinese_fonts if f not in sorted_fonts]
    sorted_fonts.extend(remaining_fonts)
    
    logger.info(f"找到 {len(sorted_fonts)} 個中文字體，優先使用: {', '.join(sorted_fonts[:3]) if sorted_fonts else '無'}")
    return sorted_fonts

def load_system_fonts() -> Tuple[bool, List[str]]:
    """嘗試載入系統字體並返回結果"""
    loaded_fonts = []
    success = False
    
    # 嘗試使用系統字體
    for font_path in get_system_font_paths():
        try:
            prop = fm.FontProperties(fname=font_path)
            font_name = prop.get_name()
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
            loaded_fonts.append(font_name)
            success = True
            logger.info(f"使用系統字體: {font_path}")
        except Exception as e:
            logger.warning(f"載入系統字體失敗: {str(e)}")
    
    return success, loaded_fonts

def download_noto_font() -> bool:
    """下載 Noto Sans TC 字體，返回是否成功"""
    try:
        # 創建字體目錄
        user_font_dir = os.path.join(os.path.expanduser("~"), ".matplotlib", "fonts", "ttf")
        os.makedirs(user_font_dir, exist_ok=True)
        
        # Noto Sans TC Regular 的直接下載鏈接
        font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansTC-Regular.otf"
        font_path = os.path.join(user_font_dir, "NotoSansTC-Regular.otf")
        
        # 只有在字體檔案不存在時才下載
        if not os.path.exists(font_path):
            logger.info(f"下載中文字體: {font_url}")
            response = requests.get(font_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(font_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"字體下載成功: {font_path}")
            matplotlib.font_manager._rebuild()
        else:
            logger.info(f"字體已存在: {font_path}")
        
        # 使用下載的字體
        prop = fm.FontProperties(fname=font_path)
        font_name = prop.get_name()
        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
        return True
    except Exception as e:
        logger.warning(f"下載字體失敗: {str(e)}")
        return False

def init_font_support():
    """初始化中文字體支援"""
    logger.info("初始化中文字體支援")
    system = platform.system()
    logger.info(f"當前作業系統: {system}")
    
    # 檢查 matplotlib 支援的字體
    chinese_fonts = get_available_chinese_fonts()
    has_chinese_font = len(chinese_fonts) > 0
    logger.info(f"系統是否已有中文字體: {has_chinese_font}")
    
    # 如果沒有找到中文字體，嘗試安裝
    if not has_chinese_font:
        # 嘗試使用系統內建字體
        success, loaded_fonts = load_system_fonts()
        
        # 如果系統字體加載失敗，嘗試下載字體
        if not success:
            download_noto_font()
            
        # 如果是 Linux 系統，嘗試安裝文泉驛字體
        if system == "Linux" and not success:
            try:
                import subprocess
                logger.info("嘗試安裝 Linux 系統文泉驛字體")
                subprocess.run(["apt-get", "update", "-y"], check=True, capture_output=True)
                subprocess.run(["apt-get", "install", "-y", "fonts-wqy-microhei", "fonts-wqy-zenhei"], 
                              check=True, capture_output=True)
                subprocess.run(["fc-cache", "-fv"], check=True, capture_output=True)
                matplotlib.font_manager._rebuild()
                logger.info("成功安裝 Linux 中文字體")
            except Exception as e:
                logger.warning(f"安裝 Linux 字體失敗: {str(e)}")
    
    # 設定 matplotlib 字體
    try:
        # 設定 sans-serif 字體族
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 更新可用中文字體列表
        chinese_fonts = get_available_chinese_fonts()
        
        if chinese_fonts:
            # 將可用的中文字體添加到 sans-serif 族最前面
            plt.rcParams['font.sans-serif'] = chinese_fonts + [f for f in plt.rcParams.get('font.sans-serif', []) if f not in BLACKLISTED_FONTS]
            logger.info(f"成功設定中文字體: {', '.join(chinese_fonts[:3])}")
        else:
            logger.warning("沒有找到可用的中文字體，將使用內嵌方式")
            plt.rcParams['svg.fonttype'] = 'none'
        
        # 修正負號顯示問題與其他設定
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        
        # 啟用字體偵錯模式 (確保使用支援中文的字體)
        if 'Arial Unicode MS' in chinese_fonts:
            plt.rcParams['font.sans-serif'].insert(0, 'Arial Unicode MS')
        matplotlib.rcParams['text.hinting'] = 'auto'
        matplotlib.rcParams['text.hinting_factor'] = 8
        
        # 記錄最終字體設定
        logger.info(f"最終字體設定: font.family={plt.rcParams['font.family']}, font.sans-serif前三個字體={plt.rcParams['font.sans-serif'][:3]}")
        
    except Exception as e:
        logger.warning(f"設定字體失敗: {str(e)}")

    logger.info("字體初始化完成")

# 在程式開始時初始化字體支援
init_font_support()

# 創建 FastMCP 伺服器
mcp = FastMCP("Wonderful Analysis Handler")

def load_csv_to_dataframe(
    csv_data: str = None,
    csv_path: str = None,
    header: int = 'infer'
) -> pd.DataFrame:
    """
    從 CSV 字串或路徑載入資料為 DataFrame。
    
    Args:
        csv_data: CSV 格式的資料
        csv_path: CSV 檔案的路徑
        header: 指定標題行，若無標題則設為 None
    
    Returns:
        pandas DataFrame
    """
    if csv_data and csv_path:
        raise ValueError("不能同時提供 csv_data 和 csv_path，請選擇一種方式。")
    if csv_path:
        df = pd.read_csv(csv_path, encoding='utf-8', header=header)
        logger.info(f"成功從路徑載入 CSV 資料，行數: {len(df)}，列數: {len(df.columns)}")
    elif csv_data:
        df = pd.read_csv(io.StringIO(csv_data), header=header)
        logger.info(f"成功載入 CSV 資料，行數: {len(df)}，列數: {len(df.columns)}")
    else:
        raise ValueError("必須提供 csv_data 或 csv_path")
    return df

@mcp.tool()
async def analyze_data(
    csv_data: str = None,
    csv_path: str = None,
    operations: List[Dict[str, Any]] = None,
    output_format: str = "json",
    header: int = 'infer',
    page: int = 1,
    page_size: int = 100,
    summary_only: bool = False
) -> Dict[str, Any]:
    """
    分析 CSV 資料，支援多種數據處理和分析操作
    
    Parameters:
        csv_data (Optional[str]): 
            CSV 格式的字串資料內容。與 csv_path 參數互斥，必須提供其中一個。
            範例: "id,name,age\n1,張三,25\n2,李四,30"
        
        csv_path (Optional[str]): 
            CSV 檔案的路徑。與 csv_data 參數互斥，必須提供其中一個。
            範例: "data/users.csv"
        
        operations (List[Dict[str, Any]]): 
            要執行的操作列表，每個操作是一個字典，必須包含 "type" 與可選的 "params" 欄位。
            如果不提供操作，則返回資料的基本統計資訊。
            範例: [{"type": "filter", "params": {"query": "`年齡` > 30"}}]
        
        output_format (str): 
            輸出格式，可選 "json" 或 "csv"。預設為 "json"。
            - "json": 返回結構化的 JSON 資料
            - "csv": 返回 CSV 格式的字串
        
        header (Optional[int]): 
            指定標題行參數，可為整數或 'infer'。
            - 'infer': 自動推斷標題行 (預設)
            - None: 資料沒有標題行，將使用自動產生的列名 (0, 1, 2...)
            - int: 使用指定行作為標題行 (0為第一行)
            
        page (int):
            分頁參數，指定要返回的頁碼，從1開始計算，預設為1
            
        page_size (int):
            分頁大小，每頁返回的最大記錄數，預設為100
            
        summary_only (bool):
            是否只返回摘要資訊而非完整資料，預設為False
            當設為True時，僅返回統計資訊和前幾筆資料的範例
    
    Returns:
        Dict[str, Any]: 分析結果，包含以下可能的欄位:
            - "data": 處理後的資料 (當output_format為"json"時)
            - "csv_data": 處理後的CSV字串 (當output_format為"csv"時)
            - "rows": 資料的行數
            - "columns": 資料的欄位名稱列表
            - "error": 若發生錯誤，則包含錯誤訊息
    
    操作類型說明:
        1. filter: 過濾資料
           參數:
           - query (str): 過濾條件，使用 pandas query 語法
             注意：必須使用 Python 風格的運算符，例如 & (而非 &&)、| (而非 ||)
             範例: "`年齡` > 30 & `性別` == '男'"
           
        2. group_by: 分組和聚合
           參數:
           - columns (List[str]): 分組列名列表
           - aggregations (Dict[str, str]): 聚合函數字典，格式為 {列名: 聚合函數}
             支援的聚合函數: "sum", "mean", "count", "min", "max", "median", "std", "var"
             範例: {"年齡": "mean", "薪資": "sum"}
           
        3. sort: 排序資料
           參數:
           - columns (List[str]): 排序列名列表
           - ascending (bool): 是否升序排序，預設為 True
           
        4. select: 選擇列
           參數:
           - columns (List[str]): 要選擇的列名列表
           
        5. transform: 轉換列
           參數:
           - transforms (Dict[str, str]): 轉換表達式字典，格式為 {列名: 表達式}
             範例: {"薪資_千元": "`薪資` / 1000", "年資": "`年齡` - `入職年齡`"}
           - auto_convert (bool): 是否自動嘗試將字串類型的數值欄位轉換為數值，預設為 True。
             當遇到如「unsupported operand type(s) for /: 'str' and 'str'」類型的錯誤時，
             會自動嘗試識別並轉換表達式中用到的字串類型欄位為數值。
           
        6. time_series: 時間序列處理
           參數:
           - date_column (str): 日期列名
           - frequency (str): 重採樣頻率，預設為 'D'
             支援的頻率: 'D'(每日), 'W'(每週), 'M'(每月), 'Q'(每季), 'Y'(每年)
           
        7. pivot: 樞紐表
           參數:
           - index (str): 索引列名
           - columns (str): 列標籤列名
           - values (str): 值列名
           
        8. shape: 返回資料的行數和列數
           無參數
           
        9. head: 返回資料集的前幾行
           參數:
           - n (int): 返回的行數，預設為 5
           
        10. tail: 返回資料集的後幾行
            參數:
            - n (int): 返回的行數，預設為 5
            
        11. custom: 執行自定義查詢
            參數:
            - query (str): 自定義查詢，使用 pandas DataFrame 方法
              現在也支援使用以下 pandas 模組函數:
              * pd.to_numeric: 將欄位轉換為數值類型
              * pd.to_datetime: 將欄位轉換為日期時間類型
              * pd.to_timedelta: 將欄位轉換為時間間隔類型
              * pd.cut/qcut: 將連續數據分箱
              * pd.get_dummies: 創建虛擬變數/one-hot編碼
              * pd.concat/merge: 資料合併操作
              * 其他多個資料處理函數
              範例:
              - "df['銷售額'] = pd.to_numeric(df['銷售額'])"
              - "pd.to_datetime(df['日期'])"
        
        12. advanced_group_by: 進階分組聚合與分析
            參數:
            - columns (List[str]): 分組欄位列表，必填
            - aggregations (Dict[str, str]): 聚合函數字典，格式為 {列名: 聚合函數}
              支援的聚合函數: "sum", "mean", "count", "min", "max", "median", "std", "var"
              範例: {"銷售額": "sum", "數量": "mean"}
            - having (str): 聚合後的過濾條件，類似SQL的HAVING子句
              範例: "`銷售額_sum` > 1000"
            - sort_by (Dict[str, str]): 排序設定，格式為 {欄位名: 順序}
              順序可為 "asc" (升序) 或 "desc" (降序)
              範例: {"銷售額_sum": "desc", "地區": "asc"}
            - top_n (int): 返回前N筆結果
              範例: 5 (返回前5筆結果)
            - compare_with (Dict[str, float]): 與基準值比較，格式為 {欄位名: 基準值}
              範例: {"平均用電量": 100.0}
            - percentage (bool): 是否計算百分比差異，預設為False
    
    範例 1: 過濾和排序資料
    ```
    {
        "csv_path": "銷售資料.csv",
        "operations": [
            {"type": "filter", "params": {"query": "`銷售額` > 1000"}},
            {"type": "sort", "params": {"columns": ["日期"], "ascending": true}}
        ],
        "output_format": "json"
    }
    ```
    
    範例 2: 分組聚合並轉換資料
    ```
    {
        "csv_data": "產品,地區,銷售額\n手機,北區,1200\n手機,南區,950\n電腦,北區,2100\n電腦,南區,1800",
        "operations": [
            {"type": "group_by", "params": {"columns": ["產品"], "aggregations": {"銷售額": "sum"}}},
            {"type": "transform", "params": {"銷售額_千元": "`銷售額` / 1000"}}
        ],
        "output_format": "json"
    }
    ```
    
    範例 3: 時間序列分析
    ```
    {
        "csv_path": "日銷售.csv",
        "operations": [
            {"type": "time_series", "params": {"date_column": "日期", "frequency": "M"}},
            {"type": "sort", "params": {"columns": ["日期"]}}
        ],
        "output_format": "json"
    }
    ```
    """
    try:
        logger.info(f"執行資料分析，操作數量: {len(operations) if operations else 0}")
        
        # 使用新函數載入資料
        df = load_csv_to_dataframe(csv_data, csv_path, header=header)
        
        # 如果沒有指定操作，則返回基本統計資訊
        if not operations:
            # 不需要指定操作，直接生成基本統計資訊
            operations = []
        
        # 執行請求的操作序列
        for i, operation in enumerate(operations):
            op_type = operation.get("type")
            params = operation.get("params", {})
            logger.info(f"執行操作 {i+1}/{len(operations)}: {op_type}")
            
            if op_type == "filter":
                # 過濾資料
                query = params.get("query")
                if query:
                    df = df.query(query)
                    logger.info(f"過濾後，行數: {len(df)}")
            
            elif op_type == "group_by":
                # 分組和聚合
                group_cols = params.get("columns", [])
                agg_dict = params.get("aggregations", {})
                if group_cols and agg_dict:
                    df = df.groupby(group_cols).agg(agg_dict).reset_index()
                    logger.info(f"分組後，行數: {len(df)}")
            
            elif op_type == "sort":
                # 排序
                sort_cols = params.get("columns", [])
                ascending = params.get("ascending", True)
                if sort_cols:
                    df = df.sort_values(by=sort_cols, ascending=ascending)
            
            elif op_type == "select":
                # 選擇列
                columns = params.get("columns", [])
                if columns:
                    df = df[columns]
                    logger.info(f"選擇列後，列數: {len(df.columns)}")
            
            elif op_type == "transform":
                # 轉換列
                transforms = params.get("transforms", {})
                auto_convert = params.get("auto_convert", True)  # 預設啟用自動類型轉換
                
                for col, expr in transforms.items():
                    try:
                        # 先嘗試直接執行表達式
                        df[col] = df.eval(expr)
                    except Exception as e:
                        # 如果失敗且啟用了自動轉換，嘗試自動轉換資料類型後再計算
                        if auto_convert and ("unsupported operand type" in str(e) or "could not convert string to float" in str(e)):
                            logger.info(f"嘗試自動轉換資料類型並執行表達式: {expr}")
                            
                            # 1. 提取表達式中的欄位 (用反引號包圍的字串)
                            import re
                            column_names = re.findall(r'`([^`]+)`', expr)
                            
                            # 2. 對這些欄位嘗試轉換為數值
                            for col_name in column_names:
                                try:
                                    if col_name in df.columns:
                                        # 檢查欄位是否包含的是字串型數字
                                        if pd.api.types.is_string_dtype(df[col_name]) and df[col_name].str.match(r'^-?\d*\.?\d+$').all():
                                            logger.info(f"自動將欄位 '{col_name}' 從字串轉換為數值")
                                            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                                except Exception as convert_error:
                                    logger.warning(f"無法轉換欄位 '{col_name}': {str(convert_error)}")
                            
                            # 3. 再次嘗試執行表達式
                            try:
                                df[col] = df.eval(expr)
                                logger.info(f"自動類型轉換後成功執行表達式: {expr}")
                            except Exception as retry_error:
                                # 如果仍然失敗，報告原始錯誤
                                logger.error(f"轉換後表達式執行仍失敗: {str(retry_error)}")
                                raise ValueError(f"表達式 '{expr}' 執行失敗: {str(e)}。自動類型轉換後仍然失敗。")
                        else:
                            # 未啟用自動轉換或者錯誤與類型無關，直接拋出原始錯誤
                            raise ValueError(f"表達式 '{expr}' 執行失敗: {str(e)}")
                            
                    logger.info(f"轉換列 {col}")
            
            elif op_type == "advanced_group_by":
                # 進階分組聚合功能
                group_cols = params.get("columns", [])
                agg_dict = params.get("aggregations", {})
                having = params.get("having", None)  # 類似SQL中的HAVING子句，用於過濾分組後的結果
                sort_by = params.get("sort_by", {})  # 排序設定，格式為{欄位名: 順序}，例如{"平均銷售額": "desc"}
                top_n = params.get("top_n", None)    # 返回前N個結果
                compare_with = params.get("compare_with", {})  # 與基準值比較，格式為{欄位名: 基準值}
                percentage = params.get("percentage", False)   # 是否計算百分比
                
                if not group_cols:
                    raise ValueError("進階分組需要提供分組欄位")
                
                # 1. 基本分組聚合
                if agg_dict:
                    logger.info(f"執行進階分組聚合: 分組欄位={group_cols}, 聚合函數={agg_dict}")
                    grouped = df.groupby(group_cols)
                    df = grouped.agg(agg_dict).reset_index()
                    logger.info(f"分組聚合後，形狀: {df.shape}")
                
                # 2. 應用Having過濾條件（類似SQL的HAVING）
                if having:
                    try:
                        # 使用query過濾
                        df = df.query(having)
                        logger.info(f"應用Having過濾後，行數: {len(df)}")
                    except Exception as e:
                        logger.error(f"應用Having過濾時出錯: {str(e)}")
                
                # 3. 與基準值比較
                if compare_with:
                    for col, reference in compare_with.items():
                        if col in df.columns:
                            # 創建比較列
                            try:
                                ref_value = float(reference)
                                df[f"{col}_差異"] = df[col] - ref_value
                                
                                if percentage:
                                    # 添加百分比差異列
                                    df[f"{col}_差異百分比"] = (df[col] - ref_value) / ref_value * 100
                                    # 格式化為帶%的字串
                                    df[f"{col}_差異百分比"] = df[f"{col}_差異百分比"].map('{:+.2f}%'.format)
                                
                                logger.info(f"添加了與基準值 {ref_value} 的比較列: {col}_差異")
                            except Exception as e:
                                logger.error(f"計算與基準值比較時出錯: {str(e)}")
                
                # 4. 排序
                if sort_by:
                    try:
                        sort_cols = []
                        sort_ascending = []
                        
                        for col, direction in sort_by.items():
                            if col in df.columns:
                                sort_cols.append(col)
                                sort_ascending.append(direction.lower() != "desc")
                        
                        if sort_cols:
                            df = df.sort_values(by=sort_cols, ascending=sort_ascending)
                            logger.info(f"依照 {sort_cols} 排序完成")
                    except Exception as e:
                        logger.error(f"排序時出錯: {str(e)}")
                
                # 5. 擷取前N筆
                if top_n and isinstance(top_n, int) and top_n > 0:
                    df = df.head(top_n)
                    logger.info(f"擷取前 {top_n} 筆資料")
            
            elif op_type == "time_series":
                # 時間序列處理
                date_col = params.get("date_column")
                freq = params.get("frequency", "D")
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.set_index(date_col).resample(freq).sum().reset_index()
                    logger.info(f"時間序列處理後，行數: {len(df)}")
            
            elif op_type == "pivot":
                # 樞紐表
                index = params.get("index")
                columns = params.get("columns")
                values = params.get("values")
                if index and columns and values:
                    df = pd.pivot_table(df, index=index, columns=columns, values=values)
                    df = df.reset_index()
                    logger.info(f"樞紐表後，形狀: {df.shape}")
            
            elif op_type == "shape":
                # 返回資料的行數和列數
                result = {"rows": len(df), "columns": len(df.columns)}
                logger.info(f"資料形狀: 行數: {len(df)}, 列數: {len(df.columns)}")
                return result
            
            elif op_type == "head":
                # 返回資料集的前幾行
                n = params.get("n", 5)  # 預設返回前 5 行
                result = df.head(n)
                if output_format == "json":
                    # 處理日期時間列以進行 JSON 序列化
                    for col in result.columns:
                        if pd.api.types.is_datetime64_any_dtype(result[col]):
                            result[col] = result[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    head_result = {
                        "data": result.to_dict(orient="records"),
                        "rows": len(result),
                        "columns": list(result.columns)
                    }
                    logger.info(f"返回資料集前 {n} 行，行數: {len(result)}")
                    # 使用 json.dumps 並設定 ensure_ascii=False
                    return json.loads(json.dumps(head_result, ensure_ascii=False))
                elif output_format == "csv":
                    csv_data = result.to_csv(index=False)
                    logger.info(f"返回資料集前 {n} 行 CSV，大小: {len(csv_data)} 位元組")
                    return {"csv_data": csv_data}
            
            elif op_type == "tail":
                # 返回資料集的後幾行
                n = params.get("n", 5)  # 預設返回後 5 行
                result = df.tail(n)
                if output_format == "json":
                    # 處理日期時間列以進行 JSON 序列化
                    for col in result.columns:
                        if pd.api.types.is_datetime64_any_dtype(result[col]):
                            result[col] = result[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    tail_result = {
                        "data": result.to_dict(orient="records"),
                        "rows": len(result),
                        "columns": list(result.columns)
                    }
                    logger.info(f"返回資料集後 {n} 行，行數: {len(result)}")
                    # 使用 json.dumps 並設定 ensure_ascii=False
                    return json.loads(json.dumps(tail_result, ensure_ascii=False))
                elif output_format == "csv":
                    csv_data = result.to_csv(index=False)
                    logger.info(f"返回資料集後 {n} 行 CSV，大小: {len(csv_data)} 位元組")
                    return {"csv_data": csv_data}
            
            elif op_type == "custom":
                # 執行自定義查詢
                query = params.get("query")
                if query:
                    try:
                        # 限制可用的方法，只允許使用 pandas DataFrame 的方法
                        # 移除已經在其他操作類型中實現的方法
                        allowed_methods = {
                            'describe', 'info', 'mean', 'median', 'min', 'max',
                            'sum', 'count', 'nunique', 'value_counts', 'drop', 'dropna', 'fillna',
                            'reset_index', 'set_index',
                            'agg', 'aggregate', 'apply', 'applymap', 'pipe',
                            'drop_duplicates', 'duplicated',
                            'rank', 'round', 'clip', 'astype', 'copy', 'isnull',
                            'notnull', 'between', 'isna', 'notna', 'any', 'all', 'abs', 'corr',
                            'cov', 'diff', 'pct_change', 'shift', 'isin', 'where', 'mask'
                        }
                        
                        # 擴展功能：允許基本資料類型轉換和處理
                        # 檢查是否使用了pandas模組功能 (例如 pd.to_numeric(df['列名']))
                        if query.startswith('pd.') or 'pd.' in query:
                            # 允許的 pandas 函數白名單
                            allowed_pd_functions = {
                                'to_numeric', 'to_datetime', 'to_timedelta', 
                                'isna', 'notna', 'isnull', 'notnull',
                                'cut', 'qcut', 'get_dummies', 'factorize',
                                'concat', 'merge', 'wide_to_long'
                            }
                            
                            # 檢查是否調用了允許的 pandas 函數
                            func_name = query.split('pd.')[1].split('(')[0].strip()
                            if func_name in allowed_pd_functions:
                                # 安全執行 pandas 函數
                                # 首先創建一個安全的局部作用域，只包含允許的對象
                                safe_locals = {'df': df, 'pd': pd}
                                try:
                                    # 使用 exec 執行代碼，並捕獲結果
                                    exec_code = f"result = {query}"
                                    exec(exec_code, {"__builtins__": {}}, safe_locals)
                                    result = safe_locals.get('result')
                                    
                                    # 檢查結果類型，並相應地處理
                                    if isinstance(result, pd.DataFrame):
                                        df = result
                                        logger.info(f"執行 pandas 函數查詢後，形狀: {df.shape}")
                                    elif isinstance(result, pd.Series):
                                        # 如果是單列操作，且結果是Series，更新對應的列
                                        # 提取可能的列名賦值模式：df['列名'] = pd.to_numeric(...)
                                        if '=' in query and query.split('=')[0].strip().startswith("df["):
                                            col_assign = query.split('=')[0].strip()
                                            col_name = col_assign[3:-1].strip().strip("'\"")
                                            if col_name in df.columns:
                                                df[col_name] = result
                                                logger.info(f"更新列 '{col_name}' 的資料類型")
                                            else:
                                                logger.warning(f"列 '{col_name}' 不存在，無法更新")
                                        else:
                                            # 其他情況將 Series 轉換為 DataFrame
                                            df = result.to_frame()
                                            logger.info(f"函數返回 Series，已轉換為 DataFrame，形狀: {df.shape}")
                                    else:
                                        # 其他結果類型
                                        logger.info(f"pandas 函數執行結果類型: {type(result).__name__}")
                                        return {"result": str(result)}
                                except Exception as e:
                                    logger.error(f"執行 pandas 函數錯誤: {str(e)}")
                                    return {"error": f"執行 pandas 函數錯誤: {str(e)}"}
                            else:
                                raise ValueError(f"不允許使用 pandas 函數: pd.{func_name}")
                        else:
                            # 原始的 DataFrame 方法處理
                            # 解析查詢，確保只使用允許的方法
                            method_name = query.split('(')[0].strip()
                            if method_name not in allowed_methods:
                                raise ValueError(f"不允許使用方法: {method_name}")
                            
                            # 使用 getattr 執行查詢，而不是 eval
                            method = getattr(df, method_name)
                            # 移除方法名稱，只保留參數部分
                            params_str = query[len(method_name):].strip()
                            if params_str.startswith('(') and params_str.endswith(')'):
                                params_str = params_str[1:-1]
                            
                            # 如果有參數，則執行帶參數的方法
                            if params_str:
                                # 使用 eval 來評估參數，但限制在安全的範圍內
                                # 這仍然有風險，但比直接 eval 整個查詢要安全得多
                                params_dict = {}
                                exec(f"params_dict = dict({params_str})", {"__builtins__": {}}, params_dict)
                                result = method(**params_dict)
                            else:
                                result = method()
                        
                            # 檢查結果類型，並相應地處理
                            if isinstance(result, pd.DataFrame):
                                # 對於 DataFrame 結果，更新 df 以便後續處理
                                df = result
                                logger.info(f"執行自定義查詢後，形狀: {df.shape}")
                            elif isinstance(result, pd.Series):
                                # 如果結果是 Series，轉換為 DataFrame
                                result_df = result.to_frame()
                                if output_format == "json":
                                    custom_result = {
                                        "data": result_df.to_dict(orient="records"),
                                        "rows": len(result_df),
                                        "columns": list(result_df.columns)
                                    }
                                    logger.info(f"返回自定義查詢結果 (Series)，行數: {len(result_df)}")
                                    # 使用 json.dumps 並設定 ensure_ascii=False
                                    return json.loads(json.dumps(custom_result, ensure_ascii=False))
                                elif output_format == "csv":
                                    csv_data = result_df.to_csv(index=False)
                                    logger.info(f"返回自定義查詢 CSV 結果 (Series)，大小: {len(csv_data)} 位元組")
                                    return {"csv_data": csv_data}
                            else:
                                # 如果結果是標量或其他類型，直接返回
                                logger.info(f"執行自定義查詢，結果類型: {type(result).__name__}")
                                return {"result": result}
                    except Exception as e:
                        logger.error(f"自定義查詢錯誤: {str(e)}")
                        return {"error": f"自定義查詢錯誤: {str(e)}"}
        
        # 根據請求的輸出格式返回結果
        if output_format == "json":
            # 修改這裡：直接返回資料集的內容，而不是統計資訊
            # 處理日期時間列以進行 JSON 序列化
            df_copy = df.copy()
            
            # 處理日期時間列
            for col in df_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 計算總頁數
            total_rows = len(df_copy)
            total_pages = (total_rows + page_size - 1) // page_size if total_rows > 0 else 1
            
            # 檢查頁碼是否有效
            if page < 1:
                page = 1
            elif page > total_pages:
                page = total_pages
            
            # 計算分頁範圍
            start_index = (page - 1) * page_size
            end_index = min(start_index + page_size, total_rows)
            
            # 取得此頁資料
            page_data = df_copy.iloc[start_index:end_index]
            
            # 建立結果字典
            result = {
                "total_rows": total_rows,
                "total_pages": total_pages,
                "current_page": page,
                "page_size": page_size,
                "columns": list(df_copy.columns)
            }
            
            # 依據要求提供完整資料或摘要
            if summary_only:
                # 提供基本統計資訊和前幾筆資料的範例
                numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
                if not numeric_cols.empty:
                    # 計算數值欄位的基本統計資訊
                    stats = df_copy[numeric_cols].describe().to_dict()
                    result["statistics"] = stats
                
                # 提供前10筆或更少的資料作為範例
                result["sample_data"] = df_copy.head(min(10, total_rows)).to_dict(orient="records")
                result["message"] = f"提供摘要資訊，總共 {total_rows} 行資料。使用 summary_only=False 以獲取完整資料。"
                logger.info(f"返回資料集摘要，總行數: {total_rows}")
            else:
                # 提供此頁的完整資料
                result["data"] = page_data.to_dict(orient="records")
                result["showing_rows"] = f"{start_index + 1}-{end_index} of {total_rows}"
                logger.info(f"返回資料集第 {page}/{total_pages} 頁，顯示第 {start_index + 1}-{end_index} 行，總行數: {total_rows}")
            
            # 使用 json.dumps 並設定 ensure_ascii=False
            return json.loads(json.dumps(result, ensure_ascii=False))
            
        elif output_format == "csv":
            # 如果要求摘要，只返回前100行
            if summary_only:
                csv_data = df.head(100).to_csv(index=False)
                logger.info(f"返回 CSV 摘要 (前100行)，大小: {len(csv_data)} 位元組，總行數: {len(df)}")
                return {
                    "csv_data": csv_data,
                    "message": f"提供前100行作為摘要，總共 {len(df)} 行資料。使用 summary_only=False 以獲取完整資料。"
                }
            else:
                # 分頁返回CSV
                total_rows = len(df)
                total_pages = (total_rows + page_size - 1) // page_size if total_rows > 0 else 1
                
                if page < 1:
                    page = 1
                elif page > total_pages:
                    page = total_pages
                
                start_index = (page - 1) * page_size
                end_index = min(start_index + page_size, total_rows)
                
                page_data = df.iloc[start_index:end_index]
                csv_data = page_data.to_csv(index=False)
                
                logger.info(f"返回 CSV 結果 (第 {page}/{total_pages} 頁)，大小: {len(csv_data)} 位元組")
                return {
                    "csv_data": csv_data,
                    "total_rows": total_rows,
                    "total_pages": total_pages,
                    "current_page": page,
                    "showing_rows": f"{start_index + 1}-{end_index} of {total_rows}"
                }
        else:
            return {"error": "不支援的輸出格式"}
    
    except Exception as e:
        logger.error(f"資料分析錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"資料分析錯誤: {str(e)}"}

@mcp.tool()
async def visualize_data(data: List[Dict[str, Any]], 
                    chart_type: str, 
                    x_column: str = None, 
                    y_column: str = None, 
                    title: str = None, 
                    color_column: str = None,
                    options: Dict[str, Any] = None) -> Image:
    """
    生成資料視覺化圖表，支援多種圖表類型與中文顯示
    
    Parameters:
        data (List[Dict[str, Any]]): 
            要視覺化的資料，每個元素是包含欄位值的字典
            範例: [{"月份": "一月", "銷售額": 100}, {"月份": "二月", "銷售額": 150}]
        
        chart_type (str): 
            圖表類型，區分大小寫，支援以下類型:
            - "line": 折線圖，適合顯示趨勢或時間序列
            - "bar": 柱狀圖，適合類別比較
            - "scatter": 散點圖，適合顯示兩個變數之間的關係
            - "pie": 圓餅圖，適合顯示比例分佈
            - "box": 箱線圖，適合顯示數據分布與離群值
            - "heatmap": 熱圖，適合顯示矩陣資料和相關性
        
        x_column (Optional[str]): 
            X軸使用的資料欄位名稱
            - 對於line, bar, scatter, box圖: X軸的欄位
            - 對於pie圖: 類別標籤欄位
            - 對於heatmap圖: 行索引欄位
        
        y_column (Optional[str]): 
            Y軸使用的資料欄位名稱
            - 對於line, bar, scatter, box圖: Y軸的欄位
            - 對於pie圖: 數值欄位
            - 對於heatmap圖: 列索引欄位
        
        title (Optional[str]): 
            圖表標題，可以使用中文
            範例: "2023年第一季銷售趨勢"
        
        color_column (Optional[str]): 
            用於分組著色的欄位名稱，用於line、bar、scatter和box圖表
            範例: "部門"或"產品類型"
        
        options (Optional[Dict[str, Any]]): 
            其他圖表選項，可包含以下選項:
            - "xlabel": X軸標籤文字
            - "ylabel": Y軸標籤文字
            - "figsize": 圖表尺寸，格式為[寬, 高]
            - "palette": 調色板名稱 (如 "Set1", "pastel", "deep")
            - "value_column": 熱圖的數值欄位 (僅用於heatmap類型)
    
    Returns:
        Image: 
            生成的圖表圖像物件，可直接在界面上顯示
    
    圖表類型詳細說明:
        1. line (折線圖):
           - 需要: x_column, y_column
           - 適用於: 時間序列資料、趨勢分析
           - 範例資料: [{"日期": "2023-01", "銷售額": 100}, {"日期": "2023-02", "銷售額": 150}]
           
        2. bar (柱狀圖):
           - 需要: x_column, y_column
           - 適用於: 類別比較、排名
           - 範例資料: [{"產品": "A產品", "銷售額": 150}, {"產品": "B產品", "銷售額": 200}]
           
        3. scatter (散點圖):
           - 需要: x_column, y_column
           - 適用於: 相關性分析、分佈模式
           - 範例資料: [{"身高": 170, "體重": 65}, {"身高": 180, "體重": 75}]
           
        4. pie (圓餅圖):
           - 需要: x_column (類別), y_column (數值)
           - 適用於: 比例分析、佔比顯示
           - 範例資料: [{"地區": "北區", "銷售額": 35}, {"地區": "南區", "銷售額": 25}]
           
        5. box (箱線圖):
           - 需要: x_column (類別), y_column (數值)
           - 適用於: 資料分佈、離群值檢測
           - 範例資料: [{"部門": "研發", "薪資": 50000}, {"部門": "行銷", "薪資": 48000}]
           
        6. heatmap (熱圖):
           - 需要: x_column, y_column, options.value_column
           - 適用於: 相關性矩陣、交叉表資料
           - 範例資料: [{"產品": "A產品", "季度": "Q1", "銷售額": 150}]
           
    範例 1: 生成銷售趨勢折線圖
    ```
    {
        "data": [
            {"月份": "一月", "銷售額": 120, "部門": "電子"},
            {"月份": "二月", "銷售額": 140, "部門": "電子"},
            {"月份": "一月", "銷售額": 90, "部門": "家電"},
            {"月份": "二月", "銷售額": 115, "部門": "家電"}
        ],
        "chart_type": "line",
        "x_column": "月份",
        "y_column": "銷售額",
        "title": "2023年首季銷售趨勢",
        "color_column": "部門"
    }
    ```
    
    範例 2: 生成地區銷售佔比圓餅圖
    ```
    {
        "data": [
            {"地區": "臺北市", "銷售額": 35},
            {"地區": "新北市", "銷售額": 25},
            {"地區": "桃園市", "銷售額": 20},
            {"地區": "其他地區", "銷售額": 20}
        ],
        "chart_type": "pie",
        "x_column": "地區",
        "y_column": "銷售額",
        "title": "各地區銷售佔比"
    }
    ```
    
    範例 3: 生成產品季度銷售熱圖
    ```
    {
        "data": [
            {"產品": "筆電", "季度": "Q1", "銷售額": 150},
            {"產品": "筆電", "季度": "Q2", "銷售額": 140},
            {"產品": "手機", "季度": "Q1", "銷售額": 200},
            {"產品": "手機", "季度": "Q2", "銷售額": 220}
        ],
        "chart_type": "heatmap",
        "x_column": "季度",
        "y_column": "產品",
        "title": "產品季度銷售熱圖",
        "options": {"value_column": "銷售額"}
    }
    ```
    """
    try:
        logger.info(f"生成視覺化，圖表類型: {chart_type}")
        
        # 檢查中文字體並確保設定正確
        chinese_fonts = get_available_chinese_fonts()
        if not chinese_fonts:
            logger.info("未檢測到中文字體，嘗試安裝中文字體...")
            try:
                await install_chinese_font()
                chinese_fonts = get_available_chinese_fonts()
                logger.info(f"安裝後中文字體數量: {len(chinese_fonts)}")
            except Exception as e:
                logger.warning(f"安裝中文字體時發生錯誤: {str(e)}")
        
        # 記錄字體設定狀態
        logger.info(f"字體家族設定: {plt.rcParams['font.family']}")
        logger.info(f"sans-serif 字體設定: {plt.rcParams['font.sans-serif'][:5]}")
        
        # 確保字體設定正確
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 重新設定字體優先順序，確保中文字體優先 
        # 首先篩選掉黑名單字體
        if plt.rcParams['font.sans-serif']:
            plt.rcParams['font.sans-serif'] = [f for f in plt.rcParams['font.sans-serif'] if f not in BLACKLISTED_FONTS]
        
        # 將確定支援中文的字體放在最前面
        win_fonts = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'SimSun', 'DFKai-SB', 'MingLiU']
        for font in reversed(win_fonts):
            if font in chinese_fonts:
                # 確保這個字體在最前面
                if font in plt.rcParams['font.sans-serif']:
                    plt.rcParams['font.sans-serif'].remove(font)
                plt.rcParams['font.sans-serif'].insert(0, font)
                logger.info(f"調整字體優先順序，將 {font} 設為首選")
        
        # 將資料轉換為 DataFrame
        df = pd.DataFrame(data)
        logger.info(f"成功載入資料，行數: {len(df)}，列數: {len(df.columns)}")
        
        # 提取 options 中的設定
        xlabel = options.get('xlabel') if options else None
        ylabel = options.get('ylabel') if options else None
        
        # 設定圖表
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # 尋找可用的中文字體屬性
        fontprops = None
        for font_name in chinese_fonts[:10]:  # 只嘗試前10個字體
            # 跳過黑名單中的字體
            if font_name in BLACKLISTED_FONTS:
                continue
                
            try:
                fontprops = fm.FontProperties(family=font_name)
                logger.info(f"嘗試使用字體: {font_name}")
                break
            except Exception as e:
                logger.warning(f"字體 {font_name} 載入失敗: {str(e)}")
        
        # 確認字體是否加載成功
        if fontprops:
            logger.info(f"最終選擇字體: {fontprops.get_name()}")
        else:
            logger.warning("未能加載理想的中文字體")
        
        # 繪製不同類型的圖表
        if chart_type == "line":
            _create_line_chart(df, x_column, y_column, color_column, fontprops)
        elif chart_type == "bar":
            _create_bar_chart(df, x_column, y_column, color_column, fontprops)
        elif chart_type == "scatter":
            _create_scatter_chart(df, x_column, y_column, color_column, fontprops)
        elif chart_type == "box":
            _create_box_chart(df, x_column, y_column, color_column, fontprops)
        elif chart_type == "heatmap":
            _create_heatmap(df, x_column, y_column, fontprops, options)
        elif chart_type == "pie":
            _create_pie_chart(df, x_column, y_column, fontprops)
        else:
            _create_unsupported_chart(chart_type)
        
        # 設定標題和軸標籤
        _set_chart_labels(title, xlabel, ylabel, fontprops)
        
        # 儲存圖表為圖片
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=200, bbox_inches='tight')
        img_buf.seek(0)
        plt.close()
        
        # 記錄圖像大小
        img_size = len(img_buf.getvalue())
        logger.info(f"生成圖像大小: {img_size} 位元組")
        
        # 返回圖像
        return Image(data=img_buf.getvalue(), format="png")
    
    except Exception as e:
        logger.error(f"視覺化錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return _create_error_image(str(e), fontprops if 'fontprops' in locals() else None)

def _create_line_chart(df, x_column, y_column, color_column, fontprops):
    """創建折線圖"""
    logger.info(f"生成折線圖，X 軸: {x_column}，Y 軸: {y_column}")
    if color_column:
        sns.lineplot(x=x_column, y=y_column, hue=color_column, data=df)
    else:
        sns.lineplot(x=x_column, y=y_column, data=df)
    
    if fontprops:
        logger.info(f"折線圖使用中文字體: {fontprops.get_name()}")
        plt.xticks(rotation=45, ha='right', fontproperties=fontprops)
        plt.yticks(fontproperties=fontprops)
        if color_column:
            plt.legend(prop=fontprops)
    else:
        plt.xticks(rotation=45, ha='right')

def _create_bar_chart(df, x_column, y_column, color_column, fontprops):
    """創建柱狀圖"""
    logger.info(f"生成柱狀圖，X 軸: {x_column}，Y 軸: {y_column}")
    if color_column:
        sns.barplot(x=x_column, y=y_column, hue=color_column, data=df)
    else:
        sns.barplot(x=x_column, y=y_column, data=df)
    
    if fontprops:
        logger.info(f"柱狀圖使用中文字體: {fontprops.get_name()}")
        plt.xticks(rotation=45, ha='right', fontproperties=fontprops)
        plt.yticks(fontproperties=fontprops)
        if color_column:
            plt.legend(prop=fontprops)
    else:
        plt.xticks(rotation=45, ha='right')

def _create_scatter_chart(df, x_column, y_column, color_column, fontprops):
    """創建散點圖"""
    logger.info(f"生成散點圖，X 軸: {x_column}，Y 軸: {y_column}")
    if color_column:
        sns.scatterplot(x=x_column, y=y_column, hue=color_column, data=df)
    else:
        sns.scatterplot(x=x_column, y=y_column, data=df)
    
    if fontprops:
        logger.info(f"散點圖使用中文字體: {fontprops.get_name()}")
        plt.xticks(fontproperties=fontprops)
        plt.yticks(fontproperties=fontprops)
        if color_column:
            plt.legend(prop=fontprops)

def _create_box_chart(df, x_column, y_column, color_column, fontprops):
    """創建箱線圖"""
    logger.info(f"生成箱線圖，X 軸: {x_column}，Y 軸: {y_column}")
    if color_column:
        sns.boxplot(x=x_column, y=y_column, hue=color_column, data=df)
    else:
        sns.boxplot(x=x_column, y=y_column, data=df)
    
    if fontprops:
        logger.info(f"箱線圖使用中文字體: {fontprops.get_name()}")
        plt.xticks(rotation=45, ha='right', fontproperties=fontprops)
        plt.yticks(fontproperties=fontprops)
        if color_column:
            plt.legend(prop=fontprops)
    else:
        plt.xticks(rotation=45, ha='right')

def _create_heatmap(df, x_column, y_column, fontprops, options):
    """創建熱圖"""
    logger.info(f"生成熱圖，X 軸: {x_column}，Y 軸: {y_column}")
    value_column = options.get("value_column", "value") if options else "value"
    pivot_table = df.pivot(index=y_column, columns=x_column, values=value_column)
    
    annot_kws = {'fontproperties': fontprops} if fontprops else {}
    if fontprops:
        logger.info(f"熱圖使用中文字體: {fontprops.get_name()}")
    
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".1f", annot_kws=annot_kws)
    
    if fontprops:
        plt.xticks(fontproperties=fontprops)
        plt.yticks(fontproperties=fontprops)
    else:
        plt.tick_params(axis='both', which='major', labelsize=10)

def _create_pie_chart(df, x_column, y_column, fontprops):
    """創建圓餅圖"""
    logger.info(f"生成圓餅圖，值: {y_column}，標籤: {x_column}")
    
    # 檢查是否提供了字體屬性，若未提供則嘗試使用常用中文字體
    if not fontprops:
        for font_name in ["Microsoft JhengHei", "Microsoft YaHei", "SimSun", "SimHei"]:
            try:
                fontprops = fm.FontProperties(family=font_name)
                logger.info(f"圓餅圖自動選用中文字體: {font_name}")
                break
            except Exception as e:
                logger.warning(f"嘗試使用字體 {font_name} 失敗: {str(e)}")
    
    # 記錄選用的字體
    if fontprops:
        logger.info(f"圓餅圖使用中文字體: {fontprops.get_name()}")
        plt.pie(df[y_column].abs(), labels=df[x_column], autopct='%1.1f%%', 
                textprops={'fontproperties': fontprops})
    else:
        logger.warning("未找到適合的中文字體，使用預設字體")
        plt.pie(df[y_column].abs(), labels=df[x_column], autopct='%1.1f%%')
    
    plt.axis('equal')
    plt.tight_layout(pad=4.0)
    
    # 如果標籤太多，使用圖例
    if len(df) > 7:
        if fontprops:
            plt.legend(loc='best', bbox_to_anchor=(0.9, 0.5), prop=fontprops)
        else:
            plt.legend(loc='best', bbox_to_anchor=(0.9, 0.5))

def _create_unsupported_chart(chart_type):
    """處理不支援的圖表類型"""
    plt.text(0.5, 0.5, f"不支援的圖表類型: {chart_type}", 
            horizontalalignment='center', verticalalignment='center',
            fontsize=12, color='red')
    plt.axis('off')

def _set_chart_labels(title, xlabel, ylabel, fontprops):
    """設定圖表標題和軸標籤"""
    if title:
        logger.info(f"設定圖表標題: {title}")
        if fontprops:
            plt.title(title, fontproperties=fontprops)
        else:
            plt.title(title)
    
    if xlabel:
        logger.info(f"設定 X 軸標籤: {xlabel}")
        if fontprops:
            plt.xlabel(xlabel, fontproperties=fontprops)
        else:
            plt.xlabel(xlabel)
    
    if ylabel:
        logger.info(f"設定 Y 軸標籤: {ylabel}")
        if fontprops:
            plt.ylabel(ylabel, fontproperties=fontprops)
        else:
            plt.ylabel(ylabel)
    
    plt.tight_layout()

def _create_error_image(error_message, fontprops=None):
    """創建錯誤訊息圖片"""
    plt.figure(figsize=(10, 6))
    
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 尋找可用的中文字體
        if not fontprops:
            for font_name in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "SimSun"]:
                try:
                    fontprops = fm.FontProperties(family=font_name)
                    logger.info(f"錯誤處理使用中文字體: {font_name}")
                    break
                except Exception:
                    continue
        
        if fontprops:
            plt.text(0.5, 0.5, f"視覺化錯誤: {error_message}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red', fontproperties=fontprops)
        else:
            plt.text(0.5, 0.5, f"視覺化錯誤: {error_message}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red')
    except Exception as e:
        logger.warning(f"在錯誤處理中設定字體失敗: {str(e)}")
        plt.text(0.5, 0.5, f"視覺化錯誤: {error_message}", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='red')
    
    plt.axis('off')
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=200, bbox_inches='tight')
    img_buf.seek(0)
    plt.close()
    
    logger.info(f"生成錯誤圖像，大小: {len(img_buf.getvalue())} 位元組")
    return Image(data=img_buf.getvalue(), format="png")

@mcp.tool()
async def advanced_statistics(
    csv_data: str,
    operations: List[Dict[str, Any]] = None,
    header: int = 'infer'
) -> Dict[str, Any]:
    """
    執行進階統計分析，提供相關性、時間序列、分佈和假設檢定等功能
    
    Parameters:
        csv_data (str): 
            CSV 格式的資料字串
            範例: "id,年齡,收入,教育程度\n1,25,35000,大學\n2,35,48000,碩士\n3,42,60000,博士"
        
        operations (List[Dict[str, Any]]): 
            要執行的統計分析列表，每個分析是一個包含type和可選params的字典
            如果不提供操作，則預設執行所有可用的分析類型
            範例: [{"type": "correlation"}, {"type": "distribution", "params": {"columns": ["年齡", "收入"]}}]
        
        header (Optional[int]): 
            CSV資料的標題行設定
            - 'infer': 自動推斷標題行 (預設)
            - None: 資料沒有標題行，使用自動產生的欄位名稱
            - int: 使用指定行索引作為標題 (0為第一行)
    
    Returns:
        Dict[str, Any]: 
            統計分析結果，包含以下可能的欄位:
            - "correlation": 相關性分析結果
            - "time_series": 時間序列分析結果
            - "distribution": 分佈分析結果
            - "hypothesis_tests": 假設檢定結果
            - "error": 若發生錯誤，則包含錯誤訊息
    
    統計分析類型詳細說明:
        1. correlation (相關性分析):
           - 功能: 計算數值變數之間的Pearson相關係數矩陣
           - 無需額外參數
           - 結果格式: {"欄位1": {"欄位1": 1.0, "欄位2": 0.7}, "欄位2": {"欄位1": 0.7, "欄位2": 1.0}}
           - 適用場景: 探索變數間的線性關係、特徵選擇、多重共線性檢查
           
        2. time_series (時間序列分析):
           - 功能: 分析時間序列資料的趨勢和週期性
           - 自動檢測日期列(包含 'date' 或 'time' 的列名)
           - 結果格式: 包含daily、weekly和monthly的時間序列聚合資料
           - 適用場景: 趨勢分析、季節性檢測、時間模式識別
           
        3. distribution (分佈分析):
           - 功能: 分析數值變數的分佈特性
           - 參數:
             * columns (List[str]): 要分析的列名列表，預設分析所有數值列
           - 結果格式: 每列包含histogram、bin_edges和分位數資訊
           - 適用場景: 資料品質檢查、離群值識別、分佈型態探索
           
        4. hypothesis_test (假設檢定):
           - 功能: 執行統計假設檢定
           - 參數:
             * t_test (Dict): t檢定參數，包含:
               - column1 (str): 第一組資料欄位
               - column2 (str): 第二組資料欄位
               - equal_variance (bool): 是否假設等方差，預設為True
             * normality_test (Dict): 正態性檢定參數，包含:
               - column (str): 要檢定的欄位
           - 結果格式: 包含檢定統計量、p值和結論
           - 適用場景: 組間差異檢定、資料分佈檢定、實驗效果評估
    
    範例 1: 執行完整的統計分析
    ```
    {
        "csv_data": "日期,產品,銷售額,客戶數\n2023-01-01,A,12000,120\n2023-01-02,A,14500,130\n2023-01-01,B,9500,95\n2023-01-02,B,11000,105",
        "operations": [
            {"type": "correlation"},
            {"type": "time_series"},
            {"type": "distribution"},
            {"type": "hypothesis_test", "params": {"normality_test": {"column": "銷售額"}}}
        ]
    }
    ```
    
    範例 2: 僅執行相關性分析和分佈分析
    ```
    {
        "csv_data": "學號,數學,英文,科學\n001,85,92,88\n002,76,85,90\n003,92,89,94\n004,68,72,75",
        "operations": [
            {"type": "correlation"},
            {"type": "distribution", "params": {"columns": ["數學", "英文", "科學"]}}
        ]
    }
    ```
    
    範例 3: 執行t檢定比較兩組資料
    ```
    {
        "csv_data": "組別,分數\nA,76\nA,82\nA,88\nA,74\nB,85\nB,92\nB,89\nB,90",
        "operations": [
            {"type": "hypothesis_test", "params": {"t_test": {"column1": "A組分數", "column2": "B組分數", "equal_variance": false}}}
        ]
    }
    ```
    """
    try:
        logger.info("執行進階統計分析")
        
        # 使用新函數載入資料
        df = load_csv_to_dataframe(csv_data, header=header)
        
        # 如果沒有指定操作，則執行所有可用的分析
        if not operations:
            operations = [
                {"type": "correlation"},
                {"type": "time_series"},
                {"type": "distribution"},
                {"type": "hypothesis_test"}
            ]
        
        # 執行進階統計分析
        results = {}
        
        for operation in operations:
            op_type = operation.get("type")
            params = operation.get("params", {})
            
            if op_type == "correlation":
                # 相關性分析
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr_matrix = numeric_df.corr().round(2)
                    # 將 NaN 轉換為 None 以進行 JSON 序列化
                    corr_dict = corr_matrix.where(~pd.isna(corr_matrix), None).to_dict()
                    results["correlation"] = corr_dict
                    logger.info(f"完成相關性分析，數值列數: {len(numeric_df.columns)}")
            
            elif op_type == "time_series":
                # 時間序列分析（如果存在日期列）
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                
                if date_cols:
                    date_col = date_cols[0]
                    logger.info(f"檢測到日期列: {date_col}")
                    
                    try:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.dropna(subset=[date_col])
                        
                        # 按日/週/月分組
                        time_series = {}
                        
                        # 每日趨勢
                        daily = df.groupby(df[date_col].dt.date).size().reset_index()
                        daily.columns = [date_col, 'count']
                        daily[date_col] = daily[date_col].astype(str)  # 轉換為字串以進行 JSON 序列化
                        time_series["daily"] = daily.to_dict(orient="records")
                        
                        # 每週趨勢
                        weekly = df.groupby(df[date_col].dt.isocalendar().week).size().reset_index()
                        weekly.columns = ['week', 'count']
                        time_series["weekly"] = weekly.to_dict(orient="records")
                        
                        # 每月趨勢
                        monthly = df.groupby(df[date_col].dt.month).size().reset_index()
                        monthly.columns = ['month', 'count']
                        time_series["monthly"] = monthly.to_dict(orient="records")
                        
                        results["time_series"] = time_series
                        logger.info("完成時間序列分析")
                    except Exception as e:
                        logger.warning(f"時間序列分析失敗: {str(e)}")
            
            elif op_type == "distribution":
                # 分佈分析
                distribution = {}
                numeric_df = df.select_dtypes(include=[np.number])
                for col in numeric_df.columns:
                    try:
                        hist, bin_edges = np.histogram(df[col].dropna(), bins=10)
                        distribution[col] = {
                            "histogram": hist.tolist(),
                            "bin_edges": bin_edges.tolist(),
                            "percentiles": {
                                "25%": float(df[col].quantile(0.25)),
                                "50%": float(df[col].quantile(0.5)),
                                "75%": float(df[col].quantile(0.75))
                            }
                        }
                    except Exception as e:
                        logger.warning(f"列 {col} 的分佈分析失敗: {str(e)}")
                
                results["distribution"] = distribution
                logger.info(f"完成分佈分析，分析列數: {len(distribution)}")
            
            elif op_type == "hypothesis_test":
                # 假設檢驗
                hypothesis_tests = {}
                
                # t 檢驗
                if "t_test" in params:
                    t_test_params = params["t_test"]
                    col1 = t_test_params.get("column1")
                    col2 = t_test_params.get("column2")
                    
                    if col1 and col2 and col1 in df.columns and col2 in df.columns:
                        try:
                            t_stat, p_value = stats.ttest_ind(
                                df[col1].dropna(), 
                                df[col2].dropna(),
                                equal_var=t_test_params.get("equal_variance", True)
                            )
                            hypothesis_tests["t_test"] = {
                                "t_statistic": float(t_stat),
                                "p_value": float(p_value),
                                "significant": bool(p_value < 0.05)
                            }
                            logger.info(f"完成 t 檢驗，列: {col1} 和 {col2}")
                        except Exception as e:
                            logger.warning(f"t 檢驗失敗: {str(e)}")
                
                # 正態性檢驗
                if "normality_test" in params:
                    normality_params = params["normality_test"]
                    col = normality_params.get("column")
                    
                    if col and col in df.columns:
                        try:
                            stat, p_value = stats.shapiro(df[col].dropna())
                            hypothesis_tests["normality_test"] = {
                                "statistic": float(stat),
                                "p_value": float(p_value),
                                "normal": bool(p_value >= 0.05)
                            }
                            logger.info(f"完成正態性檢驗，列: {col}")
                        except Exception as e:
                            logger.warning(f"正態性檢驗失敗: {str(e)}")
                
                results["hypothesis_tests"] = hypothesis_tests
        
        return results
    
    except Exception as e:
        logger.error(f"進階統計分析錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"進階統計分析錯誤: {str(e)}"}

async def install_chinese_font() -> Dict[str, Any]:
    """
    安裝中文字體用於圖表顯示
    
    Returns:
        安裝結果
    """
    try:
        logger.info("開始嘗試安裝中文字體")
        
        # 檢查是否已安裝中文字體
        chinese_fonts = get_available_chinese_fonts()
        
        if chinese_fonts:
            logger.info(f"系統已有中文字體: {', '.join(chinese_fonts[:5])}")
            return {
                "status": "已存在", 
                "message": f"系統已有中文字體: {', '.join(chinese_fonts[:5])}", 
                "fonts": chinese_fonts
            }
        
        # 創建字體目錄和初始化變數
        system = platform.system()
        user_font_dir = os.path.join(os.path.expanduser("~"), ".matplotlib", "fonts", "ttf")
        os.makedirs(user_font_dir, exist_ok=True)
        logger.info(f"創建字體目錄: {user_font_dir}")
        
        # 儲存成功安裝的字體檔案
        font_files = []
        
        # 1. 嘗試下載 Noto Sans TC 字體
        try:
            url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansTC-Regular.otf"
            font_path = os.path.join(user_font_dir, "NotoSansTC-Regular.otf")
            
            if not os.path.exists(font_path):
                logger.info(f"下載字體: {url}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(font_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"成功下載字體: {font_path}")
            else:
                logger.info(f"字體已存在: {font_path}")
            
            font_files.append(font_path)
        except Exception as e:
            logger.warning(f"下載 Noto Sans TC 字體失敗: {str(e)}")
        
        # 2. 在 Linux 上嘗試安裝額外的字體包
        if system == "Linux":
            try:
                import subprocess
                logger.info("在 Linux 上安裝文泉驛字體")
                
                subprocess.run(["apt-get", "update", "-y"], check=True, capture_output=True)
                subprocess.run(
                    ["apt-get", "install", "-y", "fonts-wqy-microhei", "fonts-wqy-zenhei"], 
                    check=True, capture_output=True
                )
                
                subprocess.run(["fc-cache", "-fv"], check=True, capture_output=True)
                logger.info("成功安裝 Linux 字體包")
            except Exception as e:
                logger.warning(f"安裝 Linux 字體包失敗: {str(e)}")
        
        # 3. 重新載入字體管理器
        logger.info("重新載入字體管理器")
        fm._get_font_family_names.cache_clear()
        matplotlib.font_manager._rebuild()
        
        # 4. 設定字體
        for font_path in font_files:
            try:
                prop = fm.FontProperties(fname=font_path)
                font_name = prop.get_name()
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
                logger.info(f"添加字體到 sans-serif 列表: {font_name}")
            except Exception as e:
                logger.warning(f"加載字體 {font_path} 失敗: {str(e)}")
        
        # 確保基本設定
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 5. 確認安裝結果
        chinese_fonts_after = get_available_chinese_fonts()
        logger.info(f"安裝後可用中文字體: {', '.join(chinese_fonts_after[:5]) if chinese_fonts_after else '無'}")
        
        # 返回結果
        return {
            "status": "成功" if chinese_fonts_after else "失敗",
            "message": f"安裝了 {len(font_files)} 個字體文件",
            "fonts_before": chinese_fonts,
            "fonts_after": chinese_fonts_after,
            "installed_files": font_files
        }
    except Exception as e:
        logger.error(f"安裝中文字體錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "錯誤", "message": str(e)}

@mcp.tool()
async def analyze_workflow(
    csv_data: str = None,
    csv_path: str = None,
    workflow_name: str = None,
    custom_workflow: List[Dict[str, Any]] = None,
    header: int = 'infer',
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    執行端到端資料分析流程，支援預定義分析模板和自訂分析流程
    
    Parameters:
        csv_data (Optional[str]): 
            CSV 格式的字串資料內容。與 csv_path 參數互斥，必須提供其中一個。
            範例: "id,product,sales\n1,A,120\n2,B,150"
        
        csv_path (Optional[str]): 
            CSV 檔案的路徑。與 csv_data 參數互斥，必須提供其中一個。
            範例: "data/sales.csv"
        
        workflow_name (str): 
            預定義的分析流程名稱，若指定此參數則忽略 custom_workflow。
            目前支援以下流程:
            - "sales_analysis": 銷售數據分析流程
            - "time_series_analysis": 時間序列分析流程
            - "correlation_analysis": 相關性分析流程
            - "customer_segmentation": 客戶分群分析流程
            - "data_quality_check": 資料品質檢查流程
        
        custom_workflow (Optional[List[Dict[str, Any]]]): 
            自訂分析流程，格式與 analyze_data 操作相同，但執行順序固定。
            若同時指定 workflow_name，則優先使用 workflow_name。
        
        header (Optional[int]): 
            CSV 資料的標題行設定，與 analyze_data 參數相同。
        
        output_format (str): 
            輸出格式，可選 "json" 或 "csv"，預設為 "json"。
    
    Returns:
        Dict[str, Any]: 
            分析結果，包含流程的所有步驟結果。
    """
    try:
        logger.info(f"執行端到端分析流程，流程名稱: {workflow_name or '自訂流程'}")
        
        # 檢查必要參數
        if not csv_data and not csv_path:
            raise ValueError("必須提供 csv_data 或 csv_path")
            
        if not workflow_name and not custom_workflow:
            raise ValueError("必須提供 workflow_name 或 custom_workflow")
        
        # 預定義的分析流程
        predefined_workflows = {
            # 銷售數據分析流程
            "sales_analysis": [
                # 1. 資料清理與準備
                {"type": "custom", "params": {"query": "df['銷售額'] = pd.to_numeric(df['銷售額'], errors='coerce')"}},
                {"type": "custom", "params": {"query": "dropna(subset=['銷售額'])"}},
                
                # 2. 按產品和地區分組分析
                {"type": "advanced_group_by", "params": {
                    "columns": ["產品", "地區"],
                    "aggregations": {"銷售額": "sum", "數量": "sum"},
                    "sort_by": {"銷售額_sum": "desc"}
                }},
                
                # 3. 增加計算列 (單價、毛利等)
                {"type": "transform", "params": {
                    "transforms": {"平均單價": "`銷售額_sum` / `數量_sum`"}
                }},
                
                # 4. 可視化結果 (通過另一個函數單獨調用)
            ],
            
            # 時間序列分析流程
            "time_series_analysis": [
                # 1. 資料清理與準備
                {"type": "custom", "params": {"query": "df['日期'] = pd.to_datetime(df['日期'], errors='coerce')"}},
                {"type": "custom", "params": {"query": "dropna(subset=['日期'])"}},
                
                # 2. 按時間聚合
                {"type": "time_series", "params": {"date_column": "日期", "frequency": "M"}},
                
                # 3. 排序並計算同比/環比變化
                {"type": "sort", "params": {"columns": ["日期"]}},
                {"type": "transform", "params": {
                    "transforms": {"環比變化": "(`銷售額` - `銷售額`.shift(1)) / `銷售額`.shift(1) * 100"}
                }}
            ],
            
            # 相關性分析流程
            "correlation_analysis": [
                # 1. 資料類型轉換
                {"type": "custom", "params": {"query": "df.select_dtypes(include=['object']).columns.tolist()"}},
                # 上面的查詢會返回所有字串類型的列，之後我們需要將數值型字串轉換為數值
                {"type": "custom", "params": {
                    "query": "df['收入'] = pd.to_numeric(df['收入'], errors='coerce')"
                }},
                {"type": "custom", "params": {
                    "query": "df['年齡'] = pd.to_numeric(df['年齡'], errors='coerce')"
                }},
                
                # 2. 去除缺失值
                {"type": "custom", "params": {"query": "dropna()"}},
                
                # 3. 計算相關性矩陣
                {"type": "custom", "params": {"query": "corr()"}}
            ],
            
            # 客戶分群分析流程
            "customer_segmentation": [
                # 1. 資料準備和清理
                {"type": "custom", "params": {"query": "df['購買頻率'] = pd.to_numeric(df['購買頻率'], errors='coerce')"}},
                {"type": "custom", "params": {"query": "df['平均消費額'] = pd.to_numeric(df['平均消費額'], errors='coerce')"}},
                {"type": "custom", "params": {"query": "df['客戶忠誠度'] = pd.to_numeric(df['客戶忠誠度'], errors='coerce')"}},
                {"type": "custom", "params": {"query": "dropna(subset=['購買頻率', '平均消費額', '客戶忠誠度'])"}},
                
                # 2. 分組分析
                {"type": "advanced_group_by", "params": {
                    "columns": ["客戶等級"],
                    "aggregations": {
                        "購買頻率": "mean", 
                        "平均消費額": "mean", 
                        "客戶忠誠度": "mean"
                    },
                    "sort_by": {"平均消費額_mean": "desc"}
                }}
            ],
            
            # 資料品質檢查流程
            "data_quality_check": [
                # 1. 統計基本資訊
                {"type": "custom", "params": {"query": "info()"}},
                {"type": "custom", "params": {"query": "describe()"}},
                
                # 2. 檢查缺失值
                {"type": "custom", "params": {"query": "isnull().sum()"}},
                
                # 3. 檢查重複值
                {"type": "custom", "params": {"query": "duplicated().sum()"}},
                
                # 4. 檢查異常值 (使用簡單的統計方法)
                {"type": "custom", "params": {
                    "query": "df.select_dtypes(include=[np.number]).apply(lambda x: ((x - x.mean()).abs() > 3*x.std()).sum())"
                }}
            ]
        }
        
        # 確定要執行的流程
        workflow = []
        if workflow_name:
            if workflow_name in predefined_workflows:
                workflow = predefined_workflows[workflow_name]
            else:
                return {"error": f"找不到預定義流程: {workflow_name}。可用的流程有: {list(predefined_workflows.keys())}"}
        else:
            workflow = custom_workflow
        
        # 載入資料
        df = load_csv_to_dataframe(csv_data, csv_path, header=header)
        logger.info(f"成功載入資料，形狀: {df.shape}")
        
        # 執行流程中的每個步驟
        results = []
        for i, step in enumerate(workflow):
            try:
                logger.info(f"執行流程步驟 {i+1}/{len(workflow)}: {step['type']}")
                
                # 直接使用 analyze_data 函數處理每個步驟
                step_result = await analyze_data(
                    csv_data=df.to_csv(index=False), 
                    operations=[step],
                    output_format=output_format,
                    header=0  # 因為我們已經從 DataFrame 生成 CSV，所以標題在第一行
                )
                
                # 更新 DataFrame 以便下一步使用
                if output_format == "json" and "data" in step_result:
                    # 從返回的 JSON 資料更新 DataFrame
                    new_df = pd.DataFrame(step_result["data"])
                    if not new_df.empty:
                        df = new_df
                
                # 添加步驟結果
                results.append({
                    "step": i+1,
                    "operation": step,
                    "result": step_result
                })
                
                logger.info(f"流程步驟 {i+1} 完成，DataFrame 形狀: {df.shape}")
                
            except Exception as e:
                logger.error(f"流程步驟 {i+1} 執行失敗: {str(e)}")
                results.append({
                    "step": i+1,
                    "operation": step,
                    "error": str(e)
                })
                # 如果這是一個關鍵步驟，我們可以選擇中斷流程
                # 這裡選擇繼續執行，以便看到更多可能的錯誤
        
        # 返回最終結果
        final_result = {
            "workflow_name": workflow_name or "custom_workflow",
            "total_steps": len(workflow),
            "executed_steps": len(results),
            "final_data_shape": {"rows": len(df), "columns": len(df.columns)},
            "steps_results": results
        }
        
        # 返回最終資料集
        if output_format == "json":
            # 處理日期時間列以進行 JSON 序列化
            df_copy = df.copy()
            for col in df_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 限制返回大小，最多返回前100行
            final_result["data"] = df_copy.head(100).to_dict(orient="records")
            if len(df) > 100:
                final_result["data_note"] = f"由於大小限制，僅顯示前100行，總行數: {len(df)}"
            
        elif output_format == "csv":
            # 限制返回大小，最多返回前500行
            csv_data = df.head(500).to_csv(index=False)
            final_result["csv_data"] = csv_data
            if len(df) > 500:
                final_result["data_note"] = f"由於大小限制，僅包含前500行，總行數: {len(df)}"
        
        logger.info(f"端到端分析流程執行完成，總步驟: {len(workflow)}，最終資料形狀: {df.shape}")
        return json.loads(json.dumps(final_result, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"端到端分析流程執行失敗: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"端到端分析流程執行失敗: {str(e)}"}

# 主程式入口
if __name__ == "__main__":
    mcp.run() 