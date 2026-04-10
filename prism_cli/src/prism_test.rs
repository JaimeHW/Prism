//! Dedicated CPython compatibility test runner for Prism.

use prism_cli::cpython_tests::{
    CliAction, SubprocessPrismExecutor, execute_cli_with_executor, help_text, parse_cli_action,
    version_text,
};
use std::process::ExitCode;

fn main() -> ExitCode {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();

    match parse_cli_action(&raw_args) {
        Ok(CliAction::Help) => {
            println!("{}", help_text());
            ExitCode::SUCCESS
        }
        Ok(CliAction::Version) => {
            println!("{}", version_text());
            ExitCode::SUCCESS
        }
        Ok(CliAction::Run(args)) => {
            match execute_cli_with_executor(&args, &SubprocessPrismExecutor) {
                Ok(report) => ExitCode::from(report.exit_code()),
                Err(err) => {
                    eprintln!("prism-test: {}", err);
                    ExitCode::from(2)
                }
            }
        }
        Err(err) => {
            eprintln!("prism-test: {}", err);
            eprintln!("{}", help_text());
            ExitCode::from(2)
        }
    }
}
