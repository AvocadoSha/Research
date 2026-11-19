# Read data
data <- read.table("k.dat", header = TRUE)

T <- data$T
k <- data$k

# Transform variables for Arrhenius plot
x <- 1000 / T          # 1000/T (K^-1)
y <- log(k)            # ln(k)

# Linear fitting
fit <- lm(y ~ x)
summary(fit)

# Extract slope
m <- coef(fit)[2]

# Gas constant
R <- 8.314  # J/mol·K

# Calculate activation energy (kJ/mol)
# IMPORTANT: because x = 1000/T
# slope m = -Ea / (R * 1000)
# => Ea = -m * R * 1000  (J/mol)
Ea <- -m * R * 1000
Ea_kJ <- Ea / 1000

# Create figure
jpeg("arrhenius_MB.jpg", width = 8, height = 6, units = "in", res = 600)

par(mgp = c(2.5, 1, 0))
par(mar = c(5, 5, 3, 2))

# Plot points
plot(x, y,
     pch = 19,
     cex = 1.5,
     col = "black",
     xlab = expression("1000/T (K"^{-1}*")"),
     ylab = "ln(k)",
     las = 1)

# Add regression line
abline(fit, col = "red", lwd = 2)

# Add Ea label (nicely positioned)
text_x <- min(x) + 0.6 * (max(x) - min(x))
text_y <- min(y) + 0.8 * (max(y) - min(y))

text(text_x, text_y,
     paste0("Ea = ", round(Ea_kJ, 2), " kJ/mol"),
     cex = 1.5)

# Add box and grid
box()
grid()

dev.off()
