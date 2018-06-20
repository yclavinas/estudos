pkgs = c("sensitivity", "MOEADr", "smoof", "ecr")
inst = lapply(pkgs, library, character.only = TRUE)

mySim <- function(r, n) {
    M <- length(r)
    X <- data.frame(matrix(runif(M * n), nrow = n))
    for (m in (1:M)) {
        rm <- r[[m]]
        X[, m] <- X[, m] * (rm$max - rm$min) + rm$min
    }
    return(X)
}


DTLZ2 <- make_vectorized_smoof(prob.name  = "DTLZ3",
                               dimensions = 40,
                               n.objectives = 3)
problem.zdt1  <- list(
    name       = "DTLZ2",
    xmin       = rep(0, 40),
    xmax       = rep(1, 40),
    m          = 3
)
decomp <- list(name = "SLD")
decomp$H <- 16
aggfun <- list(name = "AWT")
update <- list(name       = "restricted",
               UseArchive = TRUE)
update$nr = 1
scaling <- list(name = "simple")
constraint <- list(name = "none")
max.evals = 10000
stopcrit  <- list(list(name    = "maxeval",
                       maxeval = max.evals))

sens_analysis <- function(x) {
    head(x[1,])
    N <- dim(x)[1]
    res <- list()
    for (i in 1:N) {
        neighbors <- list(name    = "x",
                          T       = x[i, 1],
                          #10-40
                          delta.p = x[i, 2])#0.1~1
        
        variation <- list(
            list(name = "diffmut"),
            list(name = "binrec"),
            list(name = "binrec"),
            list(name = "truncate")
        )
        variation[[1]]$basis <- "rand"
        variation[[1]]$Phi <- NULL
        variation[[2]]$rho <- x[i, 3]
        variation[[3]]$rho <- x[i, 4]
        results.orig2 <- moead(
            problem  = problem.zdt1,
            decomp = decomp,
            aggfun = aggfun,
            neighbors = neighbors,
            update = update,
            variation = variation,
            constraint = constraint,
            scaling = scaling,
            showpars = list(show.iters = "none", showevery = 10),
            stopcrit = stopcrit
        )
        out <- list(computeHV(t(data.matrix(
            results.orig2$Y
        ))))
        res <- cbind(res, out)
    }
    return(res)
}

l1 <- list(min = 10, max = 40)
l2 <- list(min = 0.1, max = 1)
l3 <- list(min = 0, max = 1)
l4 <- list(min = 0, max = 1)


X1 <- mySim(list(l1, l2, l3, l4), n = 40)
X1[, 1] <- as.integer(X1[, 1])
X2 <- mySim(list(l1, l2, l3, l4), n = 40)
X2[, 1] <- as.integer(X2[, 1])
mem_sobolEff <- memoise::memoise(sobolEff)
cmp_mem_sobolEff <- compiler::cmpfun(mem_sobolEff)
y <-
    cmp_mem_sobolEff(
        model = sens_analysis,
        X1 = X1,
        X2 = X2,
        order = 1,
        nboot = 100
    )
png(filename = 'sobolEff_order1.png')
# print(y)
plot(y)
dev.off()

z <-
    cmp_mem_sobolEff(
        model = sens_analysis,
        X1 = X1,
        X2 = X2,
        order = 2,
        nboot = 100
    )
# print(z)
png(filename = 'sobolEff_order2.png')
plot(z)
dev.off()
end_time <- Sys.time