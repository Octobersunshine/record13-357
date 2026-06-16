import pandas as pd
from typing import List, Union, Optional


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

    def full_report(
        self,
        df: pd.DataFrame,
        row_var: str,
        col_var: str,
        margins_name: str = "合计",
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
    print("【7】完整分析报告（性别 × 满意度）：")
    report = service.full_report(df, "性别", "满意度")
    for key, value in report.items():
        print(f"\n--- {key} ---")
        print(value.round(4))


if __name__ == "__main__":
    demo()
