import io
import os
import json
import logging
import platform
import requests
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

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
    csv_data: Optional[str] = None,
    csv_path: Optional[str] = None,
    header: Optional[int] = 'infer'
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
    csv_data: Optional[str] = None,
    csv_path: Optional[str] = None,
    operations: List[Dict[str, Any]] = None,
    output_format: str = "json",
    header: Optional[int] = 'infer'
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
                for col, expr in transforms.items():
                    df[col] = df.eval(expr)
                    logger.info(f"轉換列 {col}")
            
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
            for col in df_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 直接返回資料集的內容
            result = {
                "data": df_copy.to_dict(orient="records"),
                "rows": len(df_copy),
                "columns": list(df_copy.columns)
            }
            logger.info(f"返回 JSON 結果，行數: {len(df)}")
            # 使用 json.dumps 並設定 ensure_ascii=False
            return json.loads(json.dumps(result, ensure_ascii=False))
            
        elif output_format == "csv":
            csv_data = df.to_csv(index=False)
            logger.info(f"返回 CSV 結果，大小: {len(csv_data)} 位元組")
            return {"csv_data": csv_data}
        else:
            return {"error": "不支援的輸出格式"}
    
    except Exception as e:
        logger.error(f"資料分析錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"資料分析錯誤: {str(e)}"}

@mcp.tool()
async def visualize_data(data: List[Dict[str, Any]], 
                    chart_type: str, 
                    x_column: Optional[str] = None, 
                    y_column: Optional[str] = None, 
                    title: Optional[str] = None, 
                    color_column: Optional[str] = None,
                    options: Optional[Dict[str, Any]] = None) -> Image:
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
    header: Optional[int] = 'infer'
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
                            from scipy import stats
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
                            from scipy import stats
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


if __name__ == "__main__":
    mcp.run() 