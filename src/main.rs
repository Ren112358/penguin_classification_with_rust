use std::fs::File;
use std::path::Path;

use polars::prelude::*;
use polars::frame::DataFrame;
use polars::prelude::Result as PolarResult;

fn read_csv<P: AsRef<Path>>(path: P) -> PolarResult<DataFrame> {
    // The schema below is specific to palmerpenguins dataset
    // URL: https://allisonhorst.github.io/palmerpenguins/
    let schema = Schema::new(vec![
        Field::new("species", DataType::Utf8),
        Field::new("island", DataType::Utf8),
        Field::new("culmen_length_mm", DataType::Float64),
        Field::new("culmen_depth_mm", DataType::Float64),
        Field::new("flipper_length_mm", DataType::Float64),
        Field::new("body_mass_g", DataType::Float64),
        Field::new("sex", DataType::Utf8)
    ]);

    let file = File::open(path).expect("Cannot open file.");
    CsvReader::new(file)
        .with_schema(Arc::new(schema))
        .has_header(true)
        .with_ignore_parser_errors(true)
        .finish()
}

fn main() {
    let file_path = "../data/palmerpenguins.csv";
    let df: DataFrame = read_csv(file_path).unwrap();
    println!("{:?}", df);
}
