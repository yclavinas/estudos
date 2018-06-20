suppressPackageStartupMessages(library(ecr))
library(R.utils)
setwd("~/Documents/estudos/MOEADr/R/")
sourceDirectory(".")
file.sources = list.files(pattern = "*.R")

library(plotly)
decomp <- list(name = "SLD")
decomp$H <- 16

aggfun <- list(name = "AWT")

update <- list(name       = "restricted",
               UseArchive = TRUE)
update$nr = 1

scaling <- list(name = "simple")

constraint <- list(name = "none")

max.evals = round(20100/4)
stopcrit  <- list(list(name    = "maxeval",
                       maxeval = max.evals))

ZDT1 <- MOEADr::make_vectorized_smoof(prob.name  = "DTLZ2",
                                      dimensions = 60,
                                      n.objectives = 3)
problem.zdt1  <- list(
  name       = "ZDT1",
  xmin       = rep(0, 60),
  xmax       = rep(1, 60),
  m          = 3
)

problem <- problem.zdt1

variation <- list(
  list(name = "diffmut"),
  list(name = "binrec"),
  list(name = "binrec"),
  list(name = "truncate")
)
variation[[1]]$basis <- "rand"
variation[[1]]$Phi <- NULL
variation[[2]]$rho <- 0.495
variation[[3]]$rho <- 0.899

neighbors <- list(name    = "x",
                  T       = 10,
                  delta.p = 0.909)

increments <- list(
  lhs = FALSE,
  online_RA = list(gra = NULL, dt = 20)
)
set.seed(42)
control <- MOEADr::moead(
  problem  = problem.zdt1,
  preset   = NULL,
  decomp = decomp,
  aggfun = aggfun,
  neighbors = neighbors,
  update = update,
  variation = variation,
  constraint = constraint,
  scaling = scaling,
  stopcrit = stopcrit,
  showpars = list(show.iters = "dots", showevery = 10),
  seed     = 42
)

neighbors <- list(name    = "x",
                  T       = 10,
                  delta.p = 0.909)


set.seed(42)
results.myvar2 <- moead(
  problem  = problem.zdt1,
  preset   = NULL,
  decomp = decomp,
  aggfun = aggfun,
  neighbors = neighbors,
  update = update,
  variation = variation,
  constraint = constraint,
  scaling = scaling,
  stopcrit = stopcrit,
  showpars = list(show.iters = "dots", showevery = 10),
  seed     = 42
)

neighbors <- list(name    = "x",
                  T       = 10,
                  delta.p = 0.909)


increments <- list(
  lhs = FALSE,
  online_RA = list(gra = TRUE, dt = 20)
)
set.seed(42)
results.myvar2.ORA <- moead(
  problem  = problem.zdt1,
  preset   = NULL,
  decomp = decomp,
  aggfun = aggfun,
  neighbors = neighbors,
  update = update,
  variation = variation,
  constraint = constraint,
  scaling = scaling,
  stopcrit = stopcrit,
  increments = increments,
  showpars = list(show.iters = "dots", showevery = 10),
  seed     = 42
)

neighbors <- list(name    = "x",
                  T       = 10,
                  delta.p = 0.909)


increments <- list(
  lhs = TRUE,
  online_RA = list(gra = TRUE, dt = 20)
)
set.seed(42)
old_results.myvar2.ORA.lhs <- moead(
  problem  = problem.zdt1,
  preset   = NULL,
  decomp = decomp,
  aggfun = aggfun,
  neighbors = neighbors,
  update = update,
  variation = variation,
  constraint = constraint,
  scaling = scaling,
  stopcrit = stopcrit,
  increments = increments,
  showpars = list(show.iters = "dots", showevery = 10),
  seed     = 42
)

neighbors <- list(name    = "x",
                  T       = 10,
                  delta.p = 0.909)


increments <- list(
  lhs = TRUE,
  online_RA = list(gra = NULL, dt = 20)
)
set.seed(42)
results.myvar2.lhs <- moead(
  problem  = problem.zdt1,
  preset   = NULL,
  decomp = decomp,
  aggfun = aggfun,
  neighbors = neighbors,
  update = update,
  variation = variation,
  constraint = constraint,
  scaling = scaling,
  stopcrit = stopcrit,
  increments = increments,
  showpars = list(show.iters = "dots", showevery = 10),
  seed     = 42
)

neighbors <- list(
  name    = "acga",
  T       = 14,
  LR       = 30,
  delta.p = 0.909
)


set.seed(42)
results.as_cga <- moead(#error
  problem  = problem.zdt1,
  preset   = NULL,
  decomp = decomp,
  aggfun = aggfun,
  neighbors = neighbors,
  update = update,
  variation = variation,
  constraint = constraint,
  scaling = scaling,
  stopcrit = stopcrit,
  showpars = list(show.iters = "dots", showevery = 10),
  seed     = 42
)


neighbors <- list(
  name    = "acga",
  T       = 10,
  LR       = 14,
  delta.p = 0.909
)

increments <- list(
  lhs = FALSE,
  online_RA = list(gra = TRUE, dt = 20)
)
set.seed(42)
results.as_cga.ORA <- moead(#error
  problem  = problem.zdt1,
  preset   = NULL,
  decomp = decomp,
  aggfun = aggfun,
  neighbors = neighbors,
  update = update,
  variation = variation,
  constraint = constraint,
  scaling = scaling,
  stopcrit = stopcrit,
  increments = increments,
  showpars = list(show.iters = "dots", showevery = 10),
  seed     = 42
)

# plot(results.as_cga, suppress.pause = T)


neighbors <- list(
  name    = "acga",
  T       = 10,
  LR       = 14,
  delta.p = 0.909
)


increments <- list(
  lhs = TRUE,
  online_RA = list(gra = NULL, dt = 20)
)

set.seed(42)
results.as_cga.lhs <- moead(
  problem  = problem.zdt1,
  preset   = NULL,
  decomp = decomp,
  aggfun = aggfun,
  neighbors = neighbors,
  update = update,
  variation = variation,
  constraint = constraint,
  scaling = scaling,
  stopcrit = stopcrit,
  increments = increments,
  showpars = list(show.iters = "dots", showevery = 1),
  seed     = 42
)

neighbors <- list(
  name    = "acga",
  T       = 10,
  LR       = 14,
  delta.p = 0.909
)


increments <- list(
  lhs = TRUE,
  online_RA = list(gra = TRUE, dt = 20)
)

set.seed(42)
results.as_cga.ORA.lhs <- moead(
  problem  = problem.zdt1,
  preset   = NULL,
  decomp = decomp,
  aggfun = aggfun,
  neighbors = neighbors,
  update = update,
  variation = variation,
  constraint = constraint,
  scaling = scaling,
  stopcrit = stopcrit,
  increments = increments,
  showpars = list(show.iters = "dots", showevery = 1),
  seed     = 42
)

neighbors <- list(
  name    = "acga",
  T       = 10,
  LR       = 14,
  delta.p = 0.909
)


increments <- list(
  lhs = TRUE,
  online_RA = list(gra = TRUE, dt = 20)
)

results.moead.de.ORA.lhs <- moead(problem  = problem.zdt1,
                                  preset   = preset_moead("moead.de"),
                                  decomp = decomp,
                                  stopcrit = stopcrit,
                                  increments = increments,
                                  showpars = list(show.iters = "dots", showevery = 10),
                                  seed     = 42)



generate_graph <- function(res) {
  aux <- data.frame(res$Y)

  p <-
    plot_ly(
      aux,
      x = aux$f1,
      y = aux$f2,
      z = aux$f3,
      color = aux$X1,
      colors =  c('#4AC6B7', '#1972A4')
    ) %>%
    add_markers() %>%
    layout(scene = list(
      xaxis = list(title = 'X'),
      yaxis = list(title = 'Y'),
      zaxis = list(title = 'Z')
    ))
  p
}
generate_graph_nsga <- function(res) {
  aux <- data.frame(res$pareto.front)

  p <-
    plot_ly(
      aux,
      x = aux$y1,
      y = aux$y2,
      z = aux$y3,
      color = aux$X1,
      colors =  c('#4AC6B7', '#1972A4')
    ) %>%
    add_markers() %>%
    layout(scene = list(
      xaxis = list(title = 'X'),
      yaxis = list(title = 'Y'),
      zaxis = list(title = 'Z')
    ))
  p
}




# print(generate_graph(control))
# print(generate_graph(results.myvar2))
# print(generate_graph(results.myvar2.lhs))
# print(generate_graph(results.myvar2.ORA))
# print(generate_graph(results.myvar2.ORA.lhs))
# print(generate_graph(results.as_cga))
# print(generate_graph(results.as_cga.lhs))
# print(generate_graph(results.as_cga.ORA))
# print(generate_graph(results.as_cga.ORA.lhs))
# print(generate_graph(results.moead.de))
# print(generate_graph(results.orig2))
# print(generate_graph(results.orig))
# print(generate_graph(results.moead.de.ORA.lhs))
# print(generate_graph_nsga(results.nsga2))
# print(generate_graph_nsga(results.smsemoa))


# HV -> convergency, uniformity, spread
# IGD -> convergency, uniformity, spread
# R2 -> convergency
# epsilon_indicator -> convergency
# ref.points <- c(3.367973, 2.456659, 2.882586)
ref.points <- c(11, 11, 11)
ideal <- c(0,0,0)

# summary(results.myvar2, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.myvar2$Y), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.myvar2$Y), matrix(results.myvar2$nadir))# smaller is better
emoa::unary_r2_indicator(t(results.myvar2$Y), results.myvar2$W, ideal)# smaller is better?
MOEADr::calcIGD(t(results.myvar2$Y), matrix(results.myvar2$nadir))

# summary(results.as_cga, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.as_cga$Y), ref = ref.points)
emoa::epsilon_indicator(t(results.as_cga$Y), matrix(results.myvar2$nadir)) # smaller is better
emoa::unary_r2_indicator(t(results.as_cga$Y), results.myvar2$W, ideal) # higher is better?
MOEADr::calcIGD(t(results.as_cga$Y), matrix(results.myvar2$nadir)) # higher is better


# summary(results.myvar2.ORA, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.myvar2.ORA$Y), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.myvar2.ORA$Y), matrix(results.myvar2$nadir)) # smaller is better
emoa::unary_r2_indicator(t(results.myvar2.ORA$Y), results.myvar2$W, ideal) # higher is better?
MOEADr::calcIGD(t(results.myvar2.ORA$Y), matrix(results.myvar2$nadir)) # higher is better

# summary(results.as_cga.ORA, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.as_cga.ORA$Y), ref = ref.points) # higher is better
emoa::epsilon_indicator(t(results.as_cga.ORA$Y), matrix(results.myvar2$nadir)) # smaller is better
emoa::unary_r2_indicator(t(results.as_cga.ORA$Y), results.myvar2$W, ideal) # smaller is better?
MOEADr::calcIGD(t(results.as_cga.ORA$Y), matrix(results.myvar2$nadir)) # higher is better


# summary(results.myvar2.ORA.lhs, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.myvar2.ORA.lhs$Y), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.myvar2.ORA.lhs$Y), matrix(results.myvar2$nadir)) # smaller is better
emoa::unary_r2_indicator(t(results.myvar2.ORA.lhs$Y), results.myvar2$W, ideal) # smaller is better?
MOEADr::calcIGD(t(results.myvar2.ORA.lhs$Y), matrix(results.myvar2$nadir)) # higher is better

# summary(results.as_cga.ORA.lhs, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.as_cga.ORA.lhs$Y), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.as_cga.ORA.lhs$Y), matrix(results.myvar2$nadir)) # smaller is better
emoa::unary_r2_indicator(t(results.as_cga.ORA.lhs$Y), results.myvar2$W, ideal) # smaller is better?
MOEADr::calcIGD(t(results.as_cga.ORA.lhs$Y), matrix(results.myvar2$nadir)) # higher is better

# summary(results.myvar2.lhs, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.myvar2.lhs$Y), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.myvar2.lhs$Y), matrix(results.myvar2$nadir))# smaller is better
emoa::unary_r2_indicator(t(results.myvar2.lhs$Y), results.myvar2$W, ideal)# smaller is better?
MOEADr::calcIGD(t(results.myvar2.lhs$Y), matrix(results.myvar2$nadir))# higher is better

# summary(results.as_cga.lhs, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.as_cga.lhs$Y), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.as_cga.lhs$Y), matrix(results.myvar2$nadir))# smaller is better
emoa::unary_r2_indicator(t(results.as_cga.lhs$Y), results.myvar2$W, ideal)# smaller is better?
MOEADr::calcIGD(t(results.as_cga.lhs$Y), matrix(results.myvar2$nadir))# higher is better

# summary(results.moead.de, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.moead.de$Y), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.moead.de$Y), matrix(results.myvar2$nadir))# smaller is better
emoa::unary_r2_indicator(t(results.moead.de$Y), results.myvar2$W, ideal)# smaller is better?
MOEADr::calcIGD(t(results.moead.de$Y), matrix(results.myvar2$nadir))# higher is better

# summary(results.moead.de.ORA.lhs, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.moead.de.ORA.lhs$Y), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.moead.de.ORA.lhs$Y), matrix(results.myvar2$nadir))# smaller is better
emoa::unary_r2_indicator(t(results.moead.de.ORA.lhs$Y), results.myvar2$W, ideal)# smaller is better?
MOEADr::calcIGD(t(results.moead.de.ORA.lhs$Y), matrix(results.myvar2$nadir))# higher is better

# summary(results.orig, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.orig$Y), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.orig$Y), matrix(results.myvar2$nadir))# smaller is better
emoa::unary_r2_indicator(t(results.orig$Y), results.myvar2$W, ideal)# smaller is better?
MOEADr::calcIGD(t(results.orig$Y), matrix(results.myvar2$nadir))# higher is better

# summary(results.orig2, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.orig2$Y), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.orig2$Y), matrix(results.myvar2$nadir))# smaller is better
emoa::unary_r2_indicator(t(results.orig2$Y), results.myvar2$W, ideal)# smaller is better?
MOEADr::calcIGD(t(results.orig2$Y), matrix(results.myvar2$nadir))# higher is better

# summary(results.nsga2, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.nsga2$pareto.front), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.nsga2$pareto.front), matrix(results.myvar2$nadir))# smaller is better
emoa::unary_r2_indicator(t(results.nsga2$pareto.front), results.myvar2$W, ideal)# smaller is better?
MOEADr::calcIGD(t(results.nsga2$pareto.front), matrix(results.myvar2$nadir))# higher is better

# summary(results.smsemoa, ref.point = ref.points)
emoa::dominated_hypervolume(points = t(results.smsemoa$pareto.front), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.smsemoa$pareto.front), matrix(results.myvar2$nadir))# smaller is better
emoa::unary_r2_indicator(t(results.smsemoa$pareto.front), results.myvar2$W, ideal)# smaller is better?
MOEADr::calcIGD(t(results.smsemoa$pareto.front), matrix(results.myvar2$nadir))# higher is better


# plot(results.as_cga, suppress.pause = T)
# plot(results.as_cga.lhs, suppress.pause = T)
# plot(results.as_cga.ORA, suppress.pause = T)
# plot(results.as_cga.ORA.lhs, suppress.pause = T)
# plot(results.myvar2, suppress.pause = T)
# plot(results.myvar2.lhs, suppress.pause = T)
# plot(results.myvar2.ORA, suppress.pause = T)
# plot(results.myvar2.ORA.lhs, suppress.pause = T)


# summary(control, ref.point = ref.points)
# summary(results.orig, ref.point = ref.points)
# summary(results.orig2, ref.point = ref.points)
# summary(results.moead.de, ref.point = ref.points)
