+++
title = "Rust 101: Getting Started on WSL"
date = "2026-01-01"
tags = ["rust", "wsl", "programming", "getting-started"]
categories = ["posts"]
series = ["Rust 101"]
type = "post"
draft = false
math = true
description = "A beginner's guide to installing Rust on Windows Subsystem for Linux (WSL) and writing your first 'Hello, World!' program."
+++

Welcome to Rust 101! Rust is a powerful, system-level programming language that focuses on safety, speed, and concurrency. In this post, we'll walk through the process of setting up Rust on Windows Subsystem for Linux (WSL), verifying your installation, and running your first program.

## 1. Why WSL?

For developers on Windows, **WSL (Windows Subsystem for Linux)** provides a robust Linux environment without the overhead of a virtual machine or a dual-boot setup. Rust's toolchain works seamlessly on Linux, making WSL an ideal choice for Rust development.

## 2. Installing Rust

The recommended way to install Rust is through `rustup`, a command-line tool for managing Rust versions and associated tools.

### Step 1: Install Build Essentials
Before installing Rust, you'll need a linker and some C development tools. On Ubuntu or Debian-based WSL distributions, run:

```bash
sudo apt update
sudo apt install build-essential
```

### Step 2: Install Rustup
Open your WSL terminal and run the following command to download and install `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Follow the on-screen prompts (usually, pressing `1` for the default installation is sufficient). Once finished, you'll need to restart your terminal or run:

```bash
source $HOME/.cargo/env
```

### Step 3: Verify Installation
To ensure Rust is installed correctly, check the version of the compiler:

```bash
rustc --version
```

You should see something like `rustc 1.x.y (hash date)`.

## 3. Hello, World! (The Manual Way)

Let's start by writing a simple program to see how the compiler works.

1. Create a new file named `main.rs`.
2. Add the following code:

```rust
fn main() {
    println!("Hello, world!");
}
```

3. Compile the program using `rustc`:

```bash
rustc main.rs
```

4. Run the resulting executable:

```bash
./main
```

You should see `Hello, world!` printed to your terminal.

## 4. Hello, Cargo! (The Pro Way)

While `rustc` is great for simple files, real-world Rust projects use **Cargo**, Rust’s package manager and build system. Cargo handles building your code, downloading libraries (crates), and managing dependencies.

### Creating a Project
To create a new project with Cargo, run:

```bash
cargo new hello_cargo
cd hello_cargo
```

This creates a directory named `hello_cargo` with the following structure:
- `Cargo.toml`: The configuration file for your project.
- `src/main.rs`: Where your source code lives.

### Building and Running
Inside the `hello_cargo` directory, you can build and run your project with a single command:

```bash
cargo run
```

Cargo will compile your code and run the executable. If you only want to compile without running, use `cargo build`. To just check if your code compiles without producing an executable (which is faster), use `cargo check`.

## 5. Summary

In this post, we covered:
- Setting up **WSL** for Rust development.
- Installing Rust using **rustup**.
- Compiling simple files with **rustc**.
- Managing projects with **Cargo**.

Rust has a steep learning curve, but its powerful features and helpful compiler make it a rewarding language to master. Happy coding!
