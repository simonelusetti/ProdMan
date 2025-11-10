from prettytable import PrettyTable


def build_eval_table(factor_metrics):
    if not factor_metrics:
        return "(no factor metrics)"

    class_labels = sorted(
        {
            label
            for stats in factor_metrics.values()
            for label in (stats.get("per_class") or {}).keys()
        }
    )

    table = PrettyTable()
    table.field_names = ["factor", "precision", "recall", "f1", *class_labels]
    for factor, stats in sorted(factor_metrics.items()):
        row = [
            factor,
            f"{stats.get('precision', 0.0):.4f}",
            f"{stats.get('recall', 0.0):.4f}",
            f"{stats.get('f1', 0.0):.4f}",
        ]
        per_class = stats.get("per_class") or {}
        for label in class_labels:
            f1_value = per_class.get(label, {}).get("f1", 0.0)
            row.append(f"{f1_value:.4f}")
        table.add_row(row)
    return table.get_string()
