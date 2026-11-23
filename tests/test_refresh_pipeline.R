library(shiny)

# Source from the project root without launching the app.
source("app.R", local = FALSE)

run_case <- function(scripts, failed_exit = character(), failed_launch = character()) {
  started <- character()
  list_pipeline_scripts <<- function() scripts
  start_pipeline_step <<- function(script, log_path) {
    started <<- c(started, script)
    if (script %in% failed_launch) stop(paste("launch error for", script))
    writeLines(paste("log for", script), log_path)
    code <- if (script %in% failed_exit) 1L else 0L
    list(is_alive = function() FALSE, get_exit_status = function() code,
         kill = function() NULL)
  }

  testServer(server, {
    session$setInputs(refresh_btn = 1)
    for (i in seq_along(scripts)) session$elapse(500)
    stopifnot(identical(started, scripts), !refreshing(), pipe$idx == 0)
    failures <- c(failed_exit, failed_launch)
    if (length(failures)) {
      stopifnot(length(pipe$failures) == length(failures),
                grepl("finished with", pipe$finished_msg, fixed = TRUE))
      for (script in failures) stopifnot(grepl(script, pipe$error, fixed = TRUE))
      for (script in failed_exit) stopifnot(grepl(paste("log for", script), pipe$error, fixed = TRUE))
    } else {
      stopifnot(is.null(pipe$error), grepl("successfully", pipe$finished_msg, fixed = TRUE))
    }
  })
}

scripts <- c("5a.py", "5d.py", "5e.py", "5l.py")
run_case(scripts, failed_exit = c("5d.py", "5l.py"))
run_case(scripts, failed_launch = c("5a.py", "5e.py"))
run_case(scripts)

Sys.setenv(PIPELINE_XG_TIMEOUT_SECONDS = "2")
stopifnot(pipeline_step_timeout("d_get_upcoming_xg_values.py") == 2,
          is.infinite(pipeline_step_timeout("e_apply_pca.py")))
Sys.setenv(PIPELINE_XG_TIMEOUT_SECONDS = "invalid")
stopifnot(pipeline_step_timeout("d_get_upcoming_xg_values.py") == 300)
Sys.setenv(PIPELINE_XG_TIMEOUT_SECONDS = "2")

scripts <- c("d_get_upcoming_xg_values.py", "e_apply_pca.py")
started <- character()
stalled_proc <- NULL
list_pipeline_scripts <- function() scripts
start_pipeline_step <- function(script, log_path) {
  started <<- c(started, script)
  if (identical(script, scripts[1])) {
    stalled_proc <<- processx::process$new("sleep", "60", cleanup_tree = TRUE)
    return(stalled_proc)
  }
  stopifnot(!stalled_proc$is_alive())
  list(is_alive = function() FALSE, get_exit_status = function() 0L,
       kill = function() NULL)
}
testServer(server, {
  session$setInputs(refresh_btn = 1)
  stopifnot(stalled_proc$is_alive(), identical(started, scripts[1]))
  pipe$step_start <- Sys.time() - 3
  session$elapse(500)
  session$elapse(500)
  stopifnot(!stalled_proc$is_alive(), identical(started, scripts), !refreshing(),
            grepl("timed out", pipe$error, fixed = TRUE),
            grepl("1 failed step", pipe$finished_msg, fixed = TRUE))
})

# Even if both termination methods fail and the process remains alive,
# later scripts must be attempted and the warning must not claim it stopped.
started <- character()
log_paths <- character()
kill_attempts <- 0L
start_pipeline_step <- function(script, log_path) {
  started <<- c(started, script)
  log_paths <<- c(log_paths, log_path)
  writeLines(paste("log for", script), log_path)
  if (identical(script, scripts[1])) {
    return(list(
      is_alive = function() TRUE,
      kill_tree = function() { kill_attempts <<- kill_attempts + 1L; stop("kill tree failed") },
      kill = function() { kill_attempts <<- kill_attempts + 1L; stop("kill failed") },
      wait = function(timeout) stop("wait failed")
    ))
  }
  list(is_alive = function() FALSE, get_exit_status = function() 0L,
       kill = function() NULL)
}
testServer(server, {
  session$setInputs(refresh_btn = 1)
  pipe$step_start <- Sys.time() - 3
  session$elapse(500)
  session$elapse(500)
  stopifnot(identical(started, scripts), !refreshing(), kill_attempts >= 2L,
            length(unique(log_paths)) == 2,
            grepl("could not be confirmed stopped", pipe$error, fixed = TRUE),
            grepl("may still write", pipe$error, fixed = TRUE),
            grepl("log for d_get_upcoming_xg_values.py", pipe$error, fixed = TRUE),
            grepl("1 failed step", pipe$finished_msg, fixed = TRUE))
})

for (failure_type in c("is_alive", "exit_status")) {
  started <- character()
  start_pipeline_step <- function(script, log_path) {
    started <<- c(started, script)
    if (identical(script, scripts[1])) {
      return(list(
        is_alive = function() {
          if (failure_type == "is_alive") stop("monitor failed")
          FALSE
        },
        get_exit_status = function() stop("exit status failed"),
        kill_tree = function() NULL, kill = function() NULL,
        wait = function(timeout) NULL
      ))
    }
    list(is_alive = function() FALSE, get_exit_status = function() 0L,
         kill = function() NULL)
  }
  testServer(server, {
    session$setInputs(refresh_btn = 1)
    session$elapse(500)
    session$elapse(500)
    stopifnot(identical(started, scripts), !refreshing(),
              grepl(if (failure_type == "is_alive") "could not be monitored" else "exit status unavailable",
                    pipe$error, fixed = TRUE))
  })
}
Sys.unsetenv("PIPELINE_XG_TIMEOUT_SECONDS")
cat("Refresh pipeline tests passed.\n")