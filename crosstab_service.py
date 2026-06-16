import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Union, Optional, Dict, Tuple


class CrosstabService:
    def __init__(self):
        pass

    def create_crosstab(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
        values: Optional[str] = None,
        aggfunc: Optional[str] = None,
        margins: bool = True,
        margins_name: str = "合计",
        normalize: Optional[Union[str, bool]] = False,
    ) -> pd.DataFrame:
        if row_var not in df.columns:
            raise ValueError(f"行变量 '{row_var}' 不存在于数据框中")
        if col_var not in df.columns:
            raise ValueError(f"列变量 '{col_var}' 不存在于数据框中")

        crosstab = pd.crosstab(
            index=df[row_var],
            columns=df[col_var],
            values=df[values] if values else None,
            aggfunc=aggfunc,
            margins=margins,
            margins_name=margins_name,
            normalize=normalize,
        )
        return crosstab

    def get_frequency_table(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
        margins: bool = True,
        margins_name: str = "合计",
    ) -> pd.DataFrame:
        return self.create_crosstab(
            df=df,
            row_var=row_var,
            col_var=col_var,
            margins=margins,
            margins_name=margins_name,
        )

    def get_row_marginal_distribution(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
    ) -> pd.DataFrame:
        freq_table = self.get_frequency_table(df, row_var, col_var, margins=True)
        total = freq_table.iloc[-1, -1]
        row_dist = freq_table.div(total, axis=0)
        return row_dist

    def get_column_marginal_distribution(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
    ) -> pd.DataFrame:
        freq_table = self.get_frequency_table(df, row_var, col_var, margins=True)
        col_dist = freq_table.div(freq_table.iloc[-1, :], axis=1)
        return col_dist

    def get_cell_percentage(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
        margins: bool = True,
        margins_name: str = "合计",
    ) -> pd.DataFrame:
        return self.create_crosstab(
            df=df,
            row_var=row_var,
            col_var=col_var,
            margins=margins,
            margins_name=margins_name,
            normalize="all",
        )

    def get_row_percentage(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
        margins: bool = True,
        margins_name: str = "合计",
    ) -> pd.DataFrame:
        return self.create_crosstab(
            df=df,
            row_var=row_var,
            col_var=col_var,
            margins=margins,
            margins_name=margins_name,
            normalize="index",
        )

    def get_column_percentage(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
        margins: bool = True,
        margins_name: str = "合计",
    ) -> pd.DataFrame:
        return self.create_crosstab(
            df=df,
            row_var=row_var,
            col_var=col_var,
            margins=margins,
            margins_name=margins_name,
            normalize="columns",
        )

    def chi_square_test(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
        correction: Optional[bool] = None,
        lambda_: Optional[float] = None,
    ) -> Dict[str, Union[float, int, bool, pd.DataFrame]]:
        freq_table = self.get_frequency_table(df, row_var, col_var, margins=False)
        observed = freq_table.values

        has_zero = (observed == 0).any()
        min_expected = self._get_min_expected_freq(observed)

        if correction is None:
            if observed.shape[0] == 2 and observed.shape[1] == 2:
                correction = min_expected < 5 or has_zero
            else:
                correction = False

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(
                observed,
                correction=correction,
                lambda_=lambda_,
            )
        except Exception as e:
            raise RuntimeError(f"卡方检验执行失败: {str(e)}")

        expected_df = pd.DataFrame(
            expected,
            index=freq_table.index,
            columns=freq_table.columns,
        )

        result = {
            "卡方统计量": chi2,
            "p值": p_value,
            "自由度": dof,
            "是否使用连续性校正": correction,
            "存在零频数": has_zero,
            "最小期望频数": min_expected,
            "期望频数表": expected_df,
        }
        return result

    def _get_min_expected_freq(self, observed: np.ndarray) -> float:
        row_totals = observed.sum(axis=1, keepdims=True)
        col_totals = observed.sum(axis=0, keepdims=True)
        total = observed.sum()
        expected = (row_totals @ col_totals) / total
        return expected.min()

    def fisher_exact_test(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
        alternative: str = "two-sided",
    ) -> Dict[str, Union[float, str]]:
        freq_table = self.get_frequency_table(df, row_var, col_var, margins=False)

        if freq_table.shape != (2, 2):
            raise ValueError("Fisher 精确检验仅适用于 2×2 列联表")

        table = freq_table.values
        odds_ratio, p_value = stats.fisher_exact(table, alternative=alternative)

        result = {
            "优势比": odds_ratio,
            "p值": p_value,
            "备择假设": alternative,
        }
        return result

    def recommend_test(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
    ) -> Dict[str, Union[str, bool, float]]:
        freq_table = self.get_frequency_table(df, row_var, col_var, margins=False)
        observed = freq_table.values
        n_total = observed.sum()
        n_cells = observed.size
        min_expected = self._get_min_expected_freq(observed)
        has_zero = (observed == 0).any()
        is_2x2 = observed.shape == (2, 2)

        if is_2x2:
            if n_total >= 40 and min_expected >= 5:
                recommended = "卡方检验（无需校正）"
            elif n_total >= 40 and 1 <= min_expected < 5:
                recommended = "卡方检验（连续性校正）"
            else:
                recommended = "Fisher 精确检验"
        else:
            cells_lt5 = (self._get_expected_freq(observed) < 5).sum()
            ratio_lt5 = cells_lt5 / n_cells

            if min_expected >= 5:
                recommended = "卡方检验（无需校正）"
            elif min_expected >= 1 and ratio_lt5 <= 0.2:
                recommended = "卡方检验（建议结合其他方法）"
            else:
                recommended = "Fisher 精确检验（或合并类别后卡方检验）"

        result = {
            "推荐检验方法": recommended,
            "总样本量": int(n_total),
            "最小期望频数": min_expected,
            "存在零频数": has_zero,
            "是否2x2表": is_2x2,
        }
        return result

    def _get_expected_freq(self, observed: np.ndarray) -> np.ndarray:
        row_totals = observed.sum(axis=1, keepdims=True)
        col_totals = observed.sum(axis=0, keepdims=True)
        total = observed.sum()
        return (row_totals @ col_totals) / total

    def full_report(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
        margins_name: str = "合计",
        correction: Optional[bool] = None,
    ) -> dict:
        result = {}
        result["频数表"] = self.get_frequency_table(
            df, row_var, col_var, margins=True, margins_name=margins_name
        )
        result["行百分比"] = self.get_row_percentage(
            df, row_var, col_var, margins=True, margins_name=margins_name
        )
        result["列百分比"] = self.get_column_percentage(
            df, row_var, col_var, margins=True, margins_name=margins_name
        )
        result["总百分比"] = self.get_cell_percentage(
            df, row_var, col_var, margins=True, margins_name=margins_name
        )
        result["卡方检验"] = self.chi_square_test(
            df, row_var, col_var, correction=correction
        )
        result["检验方法建议"] = self.recommend_test(df, row_var, col_var)
        return result


def demo():
    data = {
        "性别": ["男", "男", "女", "女", "男", "女", "男", "女", "女", "男", "男", "女"],
        "学历": ["本科", "硕士", "本科", "博士", "硕士", "本科", "本科", "硕士", "博士", "本科", "硕士", "本科"],
        "满意度": ["满意", "一般", "满意", "非常满意", "不满意", "满意", "一般", "满意", "非常满意", "不满意", "满意", "一般"],
    }
    df = pd.DataFrame(data)

    service = CrosstabService()

    print("=" * 60)
    print("示例数据：")
    print(df)
    print()

    print("=" * 60)
    print("【1】频数表（性别 × 学历）：")
    freq = service.get_frequency_table(df, "性别", "学历")
    print(freq)
    print()

    print("=" * 60)
    print("【2】行边际分布（每行占总数的比例）：")
    row_dist = service.get_row_marginal_distribution(df, "性别", "学历")
    print(row_dist.round(4))
    print()

    print("=" * 60)
    print("【3】列边际分布（每列占总数的比例）：")
    col_dist = service.get_column_marginal_distribution(df, "性别", "学历")
    print(col_dist.round(4))
    print()

    print("=" * 60)
    print("【4】行百分比（每行合计为100%）：")
    row_pct = service.get_row_percentage(df, "性别", "学历")
    print(row_pct.round(4))
    print()

    print("=" * 60)
    print("【5】列百分比（每列合计为100%）：")
    col_pct = service.get_column_percentage(df, "性别", "学历")
    print(col_pct.round(4))
    print()

    print("=" * 60)
    print("【6】单元格总百分比（全部合计为100%）：")
    cell_pct = service.get_cell_percentage(df, "性别", "学历")
    print(cell_pct.round(4))
    print()

    print("=" * 60)
    print("【7】卡方检验（性别 × 学历）- 含零频数场景：")
    chi2_result = service.chi_square_test(df, "性别", "学历")
    print(f"卡方统计量: {chi2_result['卡方统计量']:.4f}")
    print(f"p值: {chi2_result['p值']:.4f}")
    print(f"自由度: {chi2_result['自由度']}")
    print(f"是否使用连续性校正: {chi2_result['是否使用连续性校正']}")
    print(f"存在零频数: {chi2_result['存在零频数']}")
    print(f"最小期望频数: {chi2_result['最小期望频数']:.4f}")
    print("期望频数表:")
    print(chi2_result["期望频数表"].round(4))
    print()

    print("=" * 60)
    print("【8】检验方法建议：")
    rec = service.recommend_test(df, "性别", "学历")
    for k, v in rec.items():
        print(f"  {k}: {v}")
    print()

    print("=" * 60)
    print("【9】零频数与连续性校正对比演示：")
    data_2x2 = {
        "组别": ["A", "A", "A", "B", "B", "B", "B"],
        "结果": ["阳性", "阳性", "阴性", "阴性", "阴性", "阴性", "阳性"],
    }
    df_2x2 = pd.DataFrame(data_2x2)
    print("2×2 列联表数据：")
    print(pd.crosstab(df_2x2["组别"], df_2x2["结果"], margins=True))
    print()

    print("  -- 校正前卡方检验：")
    chi2_no_corr = service.chi_square_test(df_2x2, "组别", "结果", correction=False)
    print(f"  卡方统计量: {chi2_no_corr['卡方统计量']:.4f}")
    print(f"  p值: {chi2_no_corr['p值']:.4f}")
    print()

    print("  -- 连续性校正卡方检验（自动判断）：")
    chi2_corr = service.chi_square_test(df_2x2, "组别", "结果")
    print(f"  卡方统计量: {chi2_corr['卡方统计量']:.4f}")
    print(f"  p值: {chi2_corr['p值']:.4f}")
    print(f"  是否使用校正: {chi2_corr['是否使用连续性校正']}")
    print()

    print("  -- Fisher 精确检验：")
    fisher_result = service.fisher_exact_test(df_2x2, "组别", "结果")
    print(f"  优势比: {fisher_result['优势比']:.4f}")
    print(f"  p值: {fisher_result['p值']:.4f}")
    print()

    print("=" * 60)
    print("【10】完整分析报告（性别 × 满意度）：")
    report = service.full_report(df, "性别", "满意度")
    for key, value in report.items():
        print(f"\n--- {key} ---")
        if isinstance(value, pd.DataFrame):
            print(value.round(4))
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, pd.DataFrame):
                    print(f"  {k}:")
                    print(v.round(4).to_string().replace("\n", "\n    "))
                else:
                    print(f"  {k}: {v}")
        else:
            print(value)


if __name__ == "__main__":
    demo()
