use aegis_reborn::config::Config;
use aegis_reborn::data::{load_csv, OHLCV};
use aegis_reborn::evolution::grammar::Grammar;
use aegis_reborn::evolution::EvolutionEngine;
use criterion::{criterion_group, criterion_main, Criterion};
use std::path::Path;
use std::time::Duration;

// Helper to create a minimal but realistic test setup
fn setup_engine() -> EvolutionEngine<'static> {
    // We use 'static lifetimes here because the benchmark requires objects
    // that live for the duration of the test.
    let config: &'static Config =
        Box::leak(Box::new(Config::load(Path::new("config.toml")).unwrap()));
    let grammar: &'static Grammar = Box::leak(Box::new(
        Grammar::new(Path::new(&config.grammar_file)).unwrap(),
    ));
    let all_candles: &'static Vec<OHLCV> = Box::leak(Box::new(
        load_csv(Path::new(&config.data.file_path)).unwrap(),
    ));

    // Create a smaller slice for the benchmark to keep it fast
    let data_slice = &all_candles[0..config.ga.test_window_size * 25];

    let mut engine = EvolutionEngine::new(&config.ga, &config.metrics, grammar, data_slice);
    engine.initialize_population();
    engine
}

fn benchmark_evaluate_population(c: &mut Criterion) {
    let engine = setup_engine();

    let mut group = c.benchmark_group("EvolutionEngine Performance");
    // Set a longer measurement time because this function is slow
    group.measurement_time(Duration::from_secs(20));

    group.bench_function("evaluate_population_serial", |b| {
        // The `iter` method runs the closure multiple times and measures it.
        // `clone` is needed to reset the state for each run.
        b.iter(|| {
            let mut cloned_engine = engine.clone();
            cloned_engine.evaluate_population()
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_evaluate_population);
criterion_main!(benches);
