#!/usr/bin/env Rscript

# Automates installation/registration of the IRkernel for Jupyter.
# Usage:
#   Rscript scripts/install_r_kernel.R [--system | --user] [--display-name \"R\"]
# Defaults to per-user installation unless --system is passed.

args <- commandArgs(trailingOnly = TRUE)

option_present <- function(flag) {
  any(tolower(args) == flag)
}

value_for_option <- function(flag, default = NULL) {
  matches <- which(startsWith(tolower(args), paste0(flag, "=")))
  if (length(matches) == 0) return(default)
  sub(paste0(flag, "="), "", args[matches[length(matches)]], fixed = TRUE)
}

if (option_present("--system") && option_present("--user")) {
  stop("Specify at most one of --system or --user.")
}

install_for_user <- !option_present("--system")
display_name <- value_for_option("--display-name", "R")

setup_repos <- function() {
  repos <- getOption("repos")
  if (is.null(repos) || identical(repos, c(CRAN = "@CRAN@"))) {
    options(repos = c(CRAN = "https://cloud.r-project.org"))
  }
}

ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing package ", pkg, " ...")
    install.packages(pkg)
  } else {
    message("Package ", pkg, " already installed.")
  }
}

register_kernel <- function(display_name, install_for_user) {
  message(
    "Registering IRkernel (display name: ", display_name,
    ", scope: ", if (install_for_user) "user" else "system", ") ..."
  )
  IRkernel::installspec(
    user = install_for_user,
    displayname = display_name
  )
}

setup_repos()
ensure_package("IRkernel")

tryCatch(
  {
    register_kernel(display_name, install_for_user)
    message("IRkernel setup complete.")
  },
  warning = function(w) {
    message("IRkernel setup issued a warning: ", conditionMessage(w))
  },
  error = function(e) {
    stop("Failed to register IRkernel: ", conditionMessage(e))
  }
)
