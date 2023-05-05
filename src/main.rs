use std::fs::File;
use std::path::Path;

use polars::prelude::*;
use polars::frame::DataFrame;
use polars::prelude::Result as PolarResult;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;

fn read_csv<P: AsRef<Path>>(path: P, schema: Schema) -> PolarResult<DataFrame> {
    let file = File::open(path).expect("Cannot open file.");
    CsvReader::new(file)
        .with_schema(Arc::new(schema))
        .has_header(true)
        .with_ignore_parser_errors(true)
        .finish()
}

fn select_feature_label(
    df: &DataFrame,
    feature_columns: &Vec<&str>,
    label_columns: &Vec<&str>
    ) -> (PolarResult<DataFrame>, PolarResult<DataFrame>) {
    let features = df.select(feature_columns);
    let labels = df.select(label_columns);
    return (features, labels)
}

fn convert_features_into_matrix(df: &DataFrame) -> Result<DenseMatrix<f64>> {
    let num_rows = df.height();
    let num_cols = df.width();

    let features_array = df.to_ndarray::<Float64Type>().unwrap();
    let mut matrix: DenseMatrix<f64> = BaseMatrix::zeros(num_rows, num_cols);
    let mut col: u32 = 0;
    let mut row: u32 = 0;

    for val in features_array.iter() {
        let m_row = usize::try_from(row).unwrap();
        let m_col = usize::try_from(col).unwrap();
        matrix.set(m_row, m_col, *val);

        // Since we cannot apply "%" operator for usize in Rust,
        // We need update indices as below
        if m_col == num_cols - 1 {
            row += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    Ok(matrix)
}

fn main() {
    let file_path = "../data/palmerpenguins.csv";
    // The schema below is specific to palmerpenguins dataset
    // URL: https://allisonhorst.github.io/palmerpenguins/
    // URL: https://cran.r-project.org/web/packages/palmerpenguins/readme/README.html
    let schema = Schema::new(vec![
        Field::new("rowid", DataType::Utf8),
        Field::new("species", DataType::Utf8),
        Field::new("island", DataType::Utf8),
        Field::new("bill_length_mm", DataType::Float64),
        Field::new("bill_depth_mm", DataType::Float64),
        Field::new("flipper_length_mm", DataType::Float64),
        Field::new("body_mass_g", DataType::Float64),
        Field::new("sex", DataType::Utf8),
        Field::new("year", DataType::Utf8)
    ]);

    let df: DataFrame = read_csv(file_path, schema).unwrap();

    // Confirmed there exist rows including null by the following:
    // println!("{:?}", df.null_count());
    // Therefore, drop the rows including null.
    let df_null_dropped: DataFrame = df.drop_nulls(None).unwrap();

    // select features and labels from dataset
    let feature_columns = vec![
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"];
    let label_columns = vec!["species"];
    let (features, labels) = select_feature_label(&df_null_dropped, &feature_columns, &label_columns);

    let X = convert_features_into_matrix(&features.unwrap());
    let y = labels.unwrap().to_dummies();
}
