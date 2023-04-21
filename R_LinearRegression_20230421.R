height <- c(176, 154, 138, 196, 132, 176, 181, 169, 150, 175)
bodymass <- c(82, 49, 53, 112, 47, 69, 77, 71, 62, 78)

height
bodymass

plot(bodymass, height)


plot(bodymass, height, pch = 16, cex = 1.3, col = "blue", 
     main = "HEIGHT PLOTTED AGAINST BODY MASS", xlab = "BODY MASS (kg)", ylab = "HEIGHT (cm)")

plot(bodymass, height, xlim=c(0,200), ylim=c(0,300), pch = 16, cex = 1.3, col = "blue",
     main = "HEIGHT PLOTTED AGAINST BODY MASS", xlab = "BODY MASS (kg)", ylab = "HEIGHT (cm)")


lm(height ~ bodymass)


abline(98.0054, 0.9528)
abline(100, 2)

abline(0, 0)

abline(lm(height ~ bodymass))
