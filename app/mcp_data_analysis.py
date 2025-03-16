import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import traceback
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP, Image

# 設定日誌
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("data_analysis_mcp")

# 創建 FastMCP 伺服器
mcp = FastMCP("資料分析工具")

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
    分析 CSV 資料，支援過濾、分組、聚合等操作
    
    Args:
        csv_data: CSV 格式的資料
        csv_path: CSV 檔案的路徑
        operations: 要執行的操作列表，每個操作是一個字典，包含 type 和 params
        output_format: 輸出格式，json 或 csv
        header: 指定標題行，若無標題則設為 None
    
    Returns:
        分析結果
        
    支援的操作類型：
        1. filter: 過濾資料
           參數：
           - query: 過濾條件，使用 pandas query 語法
             注意：必須使用 Python 風格的運算符，例如 & (而非 &&)、| (而非 ||)
             例如："`欄位1` > 10 & `欄位2` == 'A'"
           
        2. group_by: 分組和聚合
           參數：
           - columns: 分組列名列表
           - aggregations: 聚合函數字典，格式為 {列名: 聚合函數}
           
        3. sort: 排序資料
           參數：
           - columns: 排序列名列表
           - ascending: 是否升序排序，預設為 True
           
        4. select: 選擇列
           參數：
           - columns: 要選擇的列名列表
           
        5. transform: 轉換列
           參數：
           - transforms: 轉換表達式字典，格式為 {列名: 表達式}
           
        6. time_series: 時間序列處理
           參數：
           - date_column: 日期列名
           - frequency: 重採樣頻率，預設為 'D'（每日）
           
        7. pivot: 樞紐表
           參數：
           - index: 索引列名
           - columns: 列標籤列名
           - values: 值列名
           
        8. shape: 返回資料的行數和列數
           無參數
           
        9. head: 返回資料集的前幾行
           參數：
           - n: 返回的行數，預設為 5
           
        10. tail: 返回資料集的後幾行
            參數：
            - n: 返回的行數，預設為 5
            
        11. custom: 執行自定義查詢
            參數：
            - query: 自定義查詢，使用 pandas DataFrame 方法
            
    使用範例：
        # 基本過濾範例
        {
            "csv_path": "data.csv",
            "operations": [
                {"type": "filter", "params": {"query": "`年齡` > 30 & `性別` == '男'"}}
            ],
            "output_format": "json"
        }
        
        # 過濾和排序範例
        {
            "csv_path": "data.csv",
            "operations": [
                {"type": "filter", "params": {"query": "`學校` == '台北市立高中' & `日期` >= '2023-01-01' & `日期` <= '2023-01-31'"}},
                {"type": "sort", "params": {"columns": ["日期"], "ascending": true}}
            ],
            "output_format": "json"
        }
        
        # 過濾、選擇列和分組範例
        {
            "csv_path": "data.csv",
            "operations": [
                {"type": "filter", "params": {"query": "`成績` > 60"}},
                {"type": "select", "params": {"columns": ["學生", "科目", "成績"]}},
                {"type": "group_by", "params": {"columns": ["科目"], "aggregations": {"成績": "mean"}}}
            ],
            "output_format": "json"
        }
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
async def visualize_data(data: List[Dict[str, Any]], chart_type: str, 
                        x_column: str, y_column: str, 
                        title: Optional[str] = None, 
                        color_column: Optional[str] = None,
                        options: Optional[Dict[str, Any]] = None) -> Image:
    """
    生成資料視覺化圖表
    
    Args:
        data: 要視覺化的資料
        chart_type: 圖表類型，支援 line, bar, scatter, pie, box, heatmap
        x_column: X 軸列名
        y_column: Y 軸列名
        title: 圖表標題
        color_column: 用於顏色分組的列名
        options: 其他圖表選項
    
    Returns:
        圖表圖像
        
    支援的圖表類型：
        1. line: 折線圖
           適用於：顯示連續資料的趨勢變化
           必要參數：
           - x_column: X 軸資料（通常是時間或連續變數）
           - y_column: Y 軸資料（數值）
           選用參數：
           - color_column: 用於分組的類別變數
           
        2. bar: 柱狀圖
           適用於：比較不同類別之間的數值
           必要參數：
           - x_column: 類別變數
           - y_column: 數值變數
           選用參數：
           - color_column: 用於分組的類別變數
           
        3. scatter: 散點圖
           適用於：顯示兩個變數之間的關係
           必要參數：
           - x_column: X 軸數值變數
           - y_column: Y 軸數值變數
           選用參數：
           - color_column: 用於分組的類別變數
           
        4. pie: 圓餅圖
           適用於：顯示部分與整體的關係
           必要參數：
           - x_column: 類別標籤
           - y_column: 數值（佔比）
           
        5. box: 箱線圖
           適用於：顯示數值分佈和離群值
           必要參數：
           - x_column: 類別變數
           - y_column: 數值變數
           選用參數：
           - color_column: 用於分組的類別變數
           
        6. heatmap: 熱圖
           適用於：顯示二維資料的強度
           必要參數：
           - x_column: 行索引
           - y_column: 列索引
           選用參數：
           - options.value_column: 熱圖值的來源列
           
    使用範例：
        {
            "data": [{"month": "一月", "sales": 100}, {"month": "二月", "sales": 150}],
            "chart_type": "bar",
            "x_column": "month",
            "y_column": "sales",
            "title": "月度銷售報表"
        }
    """
    try:
        logger.info(f"生成視覺化，圖表類型: {chart_type}")
        
        # 將資料轉換為 DataFrame
        df = pd.DataFrame(data)
        logger.info(f"成功載入資料，行數: {len(df)}，列數: {len(df.columns)}")
        
        # 設定圖表樣式
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # 根據請求的圖表類型生成視覺化
        if chart_type == "line":
            logger.info(f"生成折線圖，X 軸: {x_column}，Y 軸: {y_column}")
            if color_column:
                sns.lineplot(x=x_column, y=y_column, 
                            hue=color_column, data=df)
            else:
                sns.lineplot(x=x_column, y=y_column, data=df)
        
        elif chart_type == "bar":
            logger.info(f"生成柱狀圖，X 軸: {x_column}，Y 軸: {y_column}")
            if color_column:
                sns.barplot(x=x_column, y=y_column, 
                           hue=color_column, data=df)
            else:
                sns.barplot(x=x_column, y=y_column, data=df)
        
        elif chart_type == "scatter":
            logger.info(f"生成散點圖，X 軸: {x_column}，Y 軸: {y_column}")
            if color_column:
                sns.scatterplot(x=x_column, y=y_column, 
                               hue=color_column, data=df)
            else:
                sns.scatterplot(x=x_column, y=y_column, data=df)
        
        elif chart_type == "box":
            logger.info(f"生成箱線圖，X 軸: {x_column}，Y 軸: {y_column}")
            if color_column:
                sns.boxplot(x=x_column, y=y_column, 
                           hue=color_column, data=df)
            else:
                sns.boxplot(x=x_column, y=y_column, data=df)
        
        elif chart_type == "heatmap":
            logger.info(f"生成熱圖，X 軸: {x_column}，Y 軸: {y_column}")
            pivot_table = df.pivot(index=x_column, columns=y_column, 
                                  values=options.get("value_column", "value") if options else "value")
            sns.heatmap(pivot_table, annot=True, cmap="YlGnBu")
        
        elif chart_type == "pie":
            logger.info(f"生成圓餅圖，值: {y_column}，標籤: {x_column}")
            plt.pie(df[y_column], labels=df[x_column], autopct='%1.1f%%')
            plt.axis('equal')
        
        else:
            # 如果圖表類型不支援，返回錯誤訊息
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"不支援的圖表類型: {chart_type}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red')
            plt.axis('off')
        
        # 設定標題
        if title:
            plt.title(title)
        
        # 將圖表保存為圖像
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=100)
        img_buf.seek(0)
        plt.close()
        
        # 返回圖像
        return Image(data=img_buf.getvalue(), format="png")
    
    except Exception as e:
        logger.error(f"視覺化錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 如果發生錯誤，返回錯誤訊息
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"視覺化錯誤: {str(e)}", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='red')
        plt.axis('off')
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=100)
        img_buf.seek(0)
        plt.close()
        
        return Image(data=img_buf.getvalue(), format="png")

@mcp.tool()
async def advanced_statistics(
    csv_data: str,
    operations: List[Dict[str, Any]] = None,
    header: Optional[int] = 'infer'
) -> Dict[str, Any]:
    """
    執行進階統計分析
    
    Args:
        csv_data: CSV 格式的資料
        operations: 要執行的操作列表，每個操作是一個字典，包含 type 和 params
        header: 指定標題行，若無標題則設為 None
    
    Returns:
        進階統計分析結果
        
    支援的分析類型：
        1. correlation: 相關性分析
           功能：計算數值變數之間的相關係數矩陣
           無需額外參數
           
        2. time_series: 時間序列分析
           功能：分析時間序列資料的趨勢和季節性
           自動檢測日期列（包含 'date' 或 'time' 的列名）
           
        3. distribution: 分佈分析
           功能：分析數值變數的分佈特性
           參數：
           - columns: 要分析的列名列表，預設分析所有數值列
           
        4. hypothesis_test: 假設檢定
           功能：執行統計假設檢定
           參數：
           - test_type: 檢定類型，支援 't_test'、'chi2_test'
           - group_column: 分組列名（用於比較不同組別）
           - value_column: 數值列名（用於檢定）
           
    使用範例：
        {
            "csv_data": "...",
            "operations": [
                {"type": "correlation"},
                {"type": "distribution", "params": {"columns": ["age", "income"]}}
            ]
        }
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

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """
    健康檢查
    
    Returns:
        伺服器狀態資訊
    """
    logger.info("健康檢查請求")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@mcp.tool()
async def get_tool_help() -> Dict[str, Any]:
    """
    獲取資料分析工具的使用說明
    
    Returns:
        包含所有工具說明的字典
    """
    help_info = {
        "工具概述": "這是一個資料分析工具集，提供資料載入、分析、視覺化和進階統計功能。",
        "主要功能": [
            "analyze_data: 基本資料分析，支援過濾、分組、聚合等操作",
            "visualize_data: 資料視覺化，支援多種圖表類型",
            "advanced_statistics: 進階統計分析，包括相關性、時間序列等"
        ],
        "analyze_data": {
            "描述": "分析 CSV 資料，支援過濾、分組、聚合等操作",
            "參數": {
                "csv_data": "CSV 格式的資料",
                "csv_path": "CSV 檔案的路徑",
                "operations": "要執行的操作列表，每個操作是一個字典，包含 type 和 params",
                "output_format": "輸出格式，json 或 csv",
                "header": "指定標題行，若無標題則設為 None"
            },
            "支援的操作類型": [
                "filter: 過濾資料",
                "group_by: 分組和聚合",
                "sort: 排序資料",
                "select: 選擇列",
                "transform: 轉換列",
                "time_series: 時間序列處理",
                "pivot: 樞紐表",
                "shape: 返回資料的行數和列數",
                "head: 返回資料集的前幾行",
                "tail: 返回資料集的後幾行",
                "custom: 執行自定義查詢"
            ],
            "使用範例": {
                "基本分析": {
                    "csv_path": "data.csv",
                    "output_format": "json"
                },
                "過濾和分組": {
                    "csv_path": "data.csv",
                    "operations": [
                        {"type": "filter", "params": {"query": "`年齡` > 30 & `性別` == '男'"}}
                    ],
                    "output_format": "json"
                },
                "過濾和排序": {
                    "csv_path": "data.csv",
                    "operations": [
                        {"type": "filter", "params": {"query": "`學校` == '台北市立高中' & `日期` >= '2023-01-01' & `日期` <= '2023-01-31'"}},
                        {"type": "sort", "params": {"columns": ["日期"], "ascending": True}}
                    ],
                    "output_format": "json"
                },
                "過濾、選擇列和分組": {
                    "csv_path": "data.csv",
                    "operations": [
                        {"type": "filter", "params": {"query": "`成績` > 60"}},
                        {"type": "select", "params": {"columns": ["學生", "科目", "成績"]}},
                        {"type": "group_by", "params": {"columns": ["科目"], "aggregations": {"成績": "mean"}}}
                    ],
                    "output_format": "json"
                }
            }
        },
        "visualize_data": {
            "描述": "生成資料視覺化圖表",
            "參數": {
                "data": "要視覺化的資料",
                "chart_type": "圖表類型，支援 line, bar, scatter, pie, box, heatmap",
                "x_column": "X 軸列名",
                "y_column": "Y 軸列名",
                "title": "圖表標題",
                "color_column": "用於顏色分組的列名",
                "options": "其他圖表選項"
            },
            "支援的圖表類型": [
                "line: 折線圖",
                "bar: 柱狀圖",
                "scatter: 散點圖",
                "pie: 圓餅圖",
                "box: 箱線圖",
                "heatmap: 熱圖"
            ],
            "使用範例": {
                "柱狀圖": {
                    "data": [{"month": "一月", "sales": 100}, {"month": "二月", "sales": 150}],
                    "chart_type": "bar",
                    "x_column": "month",
                    "y_column": "sales",
                    "title": "月度銷售報表"
                }
            }
        },
        "advanced_statistics": {
            "描述": "執行進階統計分析",
            "參數": {
                "csv_data": "CSV 格式的資料",
                "operations": "要執行的操作列表，每個操作是一個字典，包含 type 和 params",
                "header": "指定標題行，若無標題則設為 None"
            },
            "支援的分析類型": [
                "correlation: 相關性分析",
                "time_series: 時間序列分析",
                "distribution: 分佈分析",
                "hypothesis_test: 假設檢定"
            ],
            "使用範例": {
                "相關性分析": {
                    "csv_data": "...",
                    "operations": [
                        {"type": "correlation"}
                    ]
                }
            }
        }
    }
    
    return help_info

if __name__ == "__main__":
    mcp.run() 