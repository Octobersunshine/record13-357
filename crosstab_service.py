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

    def cramers_v(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
        correction: Optional[bool] = None,
    ) -> Dict[str, Union[float, str]]:
        chi2_result = self.chi_square_test(
            df, row_var, col_var, correction=correction
        )
        chi2 = chi2_result["卡方统计量"]
        n = chi2_result["期望频数表"].values.sum()
        dof = chi2_result["自由度"]
        n_rows, n_cols = chi2_result["期望频数表"].shape

        k = min(n_rows, n_cols)
        v = np.sqrt(chi2 / (n * (k - 1))) if n > 0 and k > 1 else 0.0

        if correction:
            phi_sq = max(0, chi2 / n - (k - 1) / (n - 1))
            k_tilde = k - (k - 1) ** 2 / (n - 1)
            v_corrected = np.sqrt(phi_sq / (k_tilde - 1)) if k_tilde > 1 else v
        else:
            v_corrected = v

        if k == 2:
            if v < 0.1:
                strength = "极弱"
            elif v < 0.3:
                strength = "弱"
            elif v < 0.5:
                strength = "中等"
            else:
                strength = "强"
        elif k == 3:
            if v < 0.07:
                strength = "极弱"
            elif v < 0.21:
                strength = "弱"
            elif v < 0.35:
                strength = "中等"
            else:
                strength = "强"
        else:
            if v < 0.06:
                strength = "极弱"
            elif v < 0.17:
                strength = "弱"
            elif v < 0.29:
                strength = "中等"
            else:
                strength = "强"

        result = {
            "Cramér's V": float(v),
            "校正后 Cramér's V": float(v_corrected),
            "效应强度": strength,
            "是否偏差校正": bool(correction) if correction is not None else False,
        }
        return result

    def test_independence(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
        correction: Optional[bool] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Union[float, str, bool, pd.DataFrame]]:
        chi2_result = self.chi_square_test(
            df, row_var, col_var, correction=correction
        )
        cramers_result = self.cramers_v(
            df, row_var, col_var, correction=correction
        )

        p_value = chi2_result["p值"]
        is_independent = p_value > alpha

        if is_independent:
            conclusion = f"在 α={alpha} 显著性水平下，不能拒绝原假设，认为 '{row_var}' 与 '{col_var}' 相互独立"
        else:
            conclusion = f"在 α={alpha} 显著性水平下，拒绝原假设，认为 '{row_var}' 与 '{col_var}' 存在显著关联（效应强度：{cramers_result['效应强度']}）"

        result = {
            "原假设 H0": f"'{row_var}' 与 '{col_var}' 相互独立",
            "备择假设 H1": f"'{row_var}' 与 '{col_var}' 不相互独立",
            "显著性水平 α": alpha,
            "卡方统计量": chi2_result["卡方统计量"],
            "p值": chi2_result["p值"],
            "自由度": chi2_result["自由度"],
            "是否使用连续性校正": chi2_result["是否使用连续性校正"],
            "Cramér's V": cramers_result["Cramér's V"],
            "校正后 Cramér's V": cramers_result["校正后 Cramér's V"],
            "效应强度": cramers_result["效应强度"],
            "是否独立": is_independent,
            "检验结论": conclusion,
            "存在零频数": chi2_result["存在零频数"],
            "最小期望频数": chi2_result["最小期望频数"],
            "期望频数表": chi2_result["期望频数表"],
        }
        return result

    def full_report(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
        margins_name: str = "合计",
        correction: Optional[bool] = None,
        alpha: float = 0.05,
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
        result["独立性检验"] = self.test_independence(
            df, row_var, col_var, correction=correction, alpha=alpha
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
    print("【9】Cramér's V 效应量分析（性别 × 学历）：")
    cv = service.cramers_v(df, "性别", "学历")
    for k, v in cv.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print()

    print("=" * 60)
    print("【10】列联表独立性检验（性别 × 学历）：")
    indep = service.test_independence(df, "性别", "学历", alpha=0.05)
    for k, v in indep.items():
        if isinstance(v, pd.DataFrame):
            print(f"  {k}:")
            print(v.round(4).to_string().replace("\n", "\n    "))
        elif isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print()

    print("=" * 60)
    print("【11】强关联数据示例演示（显著相关）：")
    data_strong = {
        "吸烟": (["是"] * 40 + ["否"] * 10 + ["是"] * 15 + ["否"] * 35),
        "肺癌": (["是"] * 40 + ["是"] * 10 + ["否"] * 15 + ["否"] * 35),
    }
    df_strong = pd.DataFrame(data_strong)
    indep_strong = service.test_independence(df_strong, "吸烟", "肺癌")
    print(f"  频数表：")
    print(pd.crosstab(df_strong["吸烟"], df_strong["肺癌"], margins=True).to_string().replace("\n", "\n    "))
    print(f"  卡方统计量: {indep_strong['卡方统计量']:.4f}")
    print(f"  p值: {indep_strong['p值']:.6e}")
    cramers_v_value = indep_strong["Cramér's V"]
    print(f"  Cramér's V: {cramers_v_value:.4f}")
    print(f"  效应强度: {indep_strong['效应强度']}")
    print(f"  检验结论: {indep_strong['检验结论']}")
    print()

    print("=" * 60)
    print("【12】零频数与连续性校正对比演示：")
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
    print("【13】完整分析报告（性别 × 满意度）：")
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
                elif isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(value)


if __name__ == "__main__":
    demo()
