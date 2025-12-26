+++
title = "Rust 102: Building a Guessing Game"
date = "2026-01-02"
tags = ["rust", "tutorial", "guessing-game", "programming"]
categories = ["posts"]
series = ["Rust 101"]
type = "post"
draft = false
math = true
description = "A step-by-step tutorial on building a guessing game in Rust, covering variables, loops, error handling, and external crates."
+++

Welcome back to our Rust series! In [Rust 101]({{< ref "rust-101.md" >}}), we set up our environment and wrote a simple "Hello, World!" program. Now, we're going to dive deeper by building a classic beginner project: a **Guessing Game**.

This tutorial is based on the official [Rust Book's Guessing Game Tutorial](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html). By the end of this post, you'll have a working game where the computer generates a random number, and you try to guess it.

## 1. Setting Up the Project

First, let's create a new project using Cargo. Open your terminal and run:

```bash
cargo new guessing_game
cd guessing_game
```

This creates a new directory called `guessing_game` with a `Cargo.toml` file and a `src/main.rs` file.

## 2. Processing a Guess

Let's start by asking the user for input. Open `src/main.rs` and replace its contents with the following:

```rust
use std::io;

fn main() {
    println!("Guess the number!");

    println!("Please input your guess.");

    let mut guess = String::new();

    io::stdin()
        .read_line(&mut guess)
        .expect("Failed to read line");

    println!("You guessed: {guess}");
}
```

### Key Concepts:
- **`use std::io;`**: Imports the input/output library from the standard library.
- **`let mut guess`**: Creates a *mutable* variable. In Rust, variables are immutable by default. Adding `mut` allows us to change the value.
- **`String::new()`**: Creates a new, empty string instance.
- **`read_line(&mut guess)`**: Takes user input and appends it to the `guess` string. The `&` indicates a reference, which allows code to access data without copying it.
- **`.expect(...)`**: Handles potential errors. If `read_line` fails, the program will crash and display the message.

Run the program with `cargo run` to test it out!

## 3. Generating a Secret Number

To make this a game, we need a secret number. Rust's standard library doesn't include random number generation, so we'll use an external crate called `rand`.

### Adding the Dependency

Open `Cargo.toml` and add `rand` to the `[dependencies]` section:

```toml
[dependencies]
rand = "0.8.5"
```

Now, update `src/main.rs` to generate a random number:

```rust
use std::io;
use rand::Rng; // Import the Rng trait

fn main() {
    println!("Guess the number!");

    let secret_number = rand::thread_rng().gen_range(1..=100);

    println!("The secret number is: {secret_number}"); // For testing

    println!("Please input your guess.");

    // ... (rest of the code)
}
```

- **`rand::thread_rng()`**: Gives us a random number generator local to the current thread.
- **`gen_range(1..=100)`**: Generates a number between 1 and 100 (inclusive).

## 4. Comparing the Guess

Now we need to compare the user's guess with the secret number. We'll use the `std::cmp::Ordering` enum and a `match` expression.

Update `src/main.rs`:

```rust
use std::cmp::Ordering;
use std::io;
use rand::Rng;

fn main() {
    // ... (generation code)

    // ... (input code)

    // Convert the String to a number
    let guess: u32 = guess.trim().parse().expect("Please type a number!");

    println!("You guessed: {guess}");

    match guess.cmp(&secret_number) {
        Ordering::Less => println!("Too small!"),
        Ordering::Greater => println!("Too big!"),
        Ordering::Equal => println!("You win!"),
    }
}
```

### Key Concepts:
- **Shadowing**: We create a new variable `guess` (type `u32`) that shadows the previous `String` variable. This is common in Rust for type conversion.
- **`parse()`**: Parses a string into a number.
- **`match`**: A powerful control flow operator that allows you to compare a value against a series of patterns.

## 5. Looping and Final Touches

The game currently ends after one guess. Let's add a loop to allow multiple guesses and handle invalid input gracefully.

Here is the final code for `src/main.rs`:

```rust
use std::cmp::Ordering;
use std::io;
use rand::Rng;

fn main() {
    println!("Guess the number!");

    let secret_number = rand::thread_rng().gen_range(1..=100);

    loop {
        println!("Please input your guess.");

        let mut guess = String::new();

        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

        // Handle invalid input
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };

        println!("You guessed: {guess}");

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break; // Exit the loop
            }
        }
    }
}
```

### What's New:
- **`loop`**: Creates an infinite loop.
- **`break`**: Exits the loop when the user wins.
- **Error Handling**: Instead of crashing on invalid input, we use `match` on the `parse()` result. If it's an error (`Err`), we `continue` to the next iteration of the loop.

## Conclusion

Congratulations! You've built a complete Guessing Game in Rust. You've learned about:
- Variables and mutability (`let`, `let mut`)
- Input/Output (`std::io`)
- External crates (`rand`)
- Pattern matching (`match`)
- Loops and control flow

In the next post, we'll explore common programming concepts in Rust in more detail. Happy coding!
