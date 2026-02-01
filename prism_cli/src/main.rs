//! Prism: Hyper-performant Python runtime.

fn main() {
    println!("Prism Python Runtime v{}", prism_core::VERSION);
    println!(
        "Python {}.{}.{} compatible",
        prism_core::PYTHON_VERSION.0,
        prism_core::PYTHON_VERSION.1,
        prism_core::PYTHON_VERSION.2,
    );
}
