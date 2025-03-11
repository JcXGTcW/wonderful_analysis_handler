import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP, Image, Context

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

@mcp.tool()
async def analyze_data(csv_data: str, operations: List[Dict[str, Any]] = None, output_format: str = "json") -> Dict[str, Any]:
    """
    分析 CSV 資料，支援過濾、分組、聚合等操作
    
    Args:
        csv_data: CSV 格式的資料
        operations: 要執行的操作列表，每個操作是一個字典，包含 type 和 params
        output_format: 輸出格式，json 或 csv
    
    Returns:
        分析結果
    """
    try:
        logger.info(f"執行資料分析，操作數量: {len(operations) if operations else 0}")
        
        # 將 CSV 字串轉換為 DataFrame
        df = pd.read_csv(io.StringIO(csv_data))
        logger.info(f"成功載入 CSV 資料，行數: {len(df)}，列數: {len(df.columns)}")
        
        # 如果沒有指定操作，則返回基本統計資訊
        if not operations:
            operations = [{"type": "summary"}]
        
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
            
            elif op_type == "summary":
                # 不需要做任何事，只是為了生成基本統計資訊
                pass
            
            elif op_type == "correlation":
                # 不需要做任何事，只是為了生成相關性矩陣
                pass
            
            elif op_type == "missing":
                # 不需要做任何事，只是為了生成缺失值分析
                pass
            
            elif op_type == "unique":
                # 不需要做任何事，只是為了生成唯一值分析
                pass
            
            elif op_type == "custom":
                # 執行自定義查詢
                query = params.get("query")
                if query:
                    try:
                        result = eval(f"df.{query}")
                        if isinstance(result, pd.DataFrame):
                            df = result
                        logger.info(f"執行自定義查詢後，形狀: {df.shape}")
                    except Exception as e:
                        logger.error(f"自定義查詢錯誤: {str(e)}")
        
        # 生成基本統計資訊
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else None
            }
        
        # 生成缺失值分析
        missing = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing[col] = {
                "count": int(missing_count),
                "percentage": float(missing_count / len(df) * 100) if len(df) > 0 else 0
            }
        
        # 生成唯一值分析
        unique = {}
        for col in df.columns:
            unique_count = df[col].nunique()
            unique[col] = {
                "count": int(unique_count),
                "percentage": float(unique_count / len(df) * 100) if len(df) > 0 else 0
            }
        
        # 根據請求的輸出格式返回結果
        if output_format == "json":
            # 處理日期時間列以進行 JSON 序列化
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            result = {
                "data": df.to_dict(orient="records"),
                "stats": stats,
                "missing": missing,
                "unique": unique,
                "rows": len(df),
                "columns": list(df.columns)
            }
            logger.info(f"返回 JSON 結果，行數: {len(df)}")
            return result
            
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
async def advanced_statistics(csv_data: str, operations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    執行進階統計分析
    
    Args:
        csv_data: CSV 格式的資料
        operations: 要執行的操作列表，每個操作是一個字典，包含 type 和 params
    
    Returns:
        進階統計分析結果
    """
    try:
        logger.info("執行進階統計分析")
        
        # 將 CSV 字串轉換為 DataFrame
        df = pd.read_csv(io.StringIO(csv_data))
        logger.info(f"成功載入 CSV 資料，行數: {len(df)}，列數: {len(df.columns)}")
        
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

if __name__ == "__main__":
    mcp.run() 