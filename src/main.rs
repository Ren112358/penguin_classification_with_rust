use std::fs::File;
use std::path::Path;

use polars::prelude::*;
use polars::frame::DataFrame;
use polars::prelude::Result as PolarResult;

fn read_csv<P: AsRef<Path>>(path: P, schema: Schema) -> PolarResult<DataFrame> {
    let file = File::open(path).expect("Cannot open file.");
    CsvReader::new(file)
        .with_schema(Arc::new(schema))
        .has_header(true)
        .with_ignore_parser_errors(true)
        .finish()
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
    //println!("{:?}", df);

    // Confirmed there exist rows including null by the following:
    // println!("{:?}", df.null_count());
    // Therefore, drop the rows including null.
    let df_null_dropped: DataFrame = df.drop_nulls(None).unwrap();
    println!("{:?}", df_null_dropped);
    println!("{:?}", df_null_dropped.null_count());
}
