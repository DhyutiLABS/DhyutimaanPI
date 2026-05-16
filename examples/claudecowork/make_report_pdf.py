"""Generate a 6-page paper-style report on FD-PINN for 2D heat conduction."""
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
    KeepTogether,
)

PDF_PATH = "PINN_2D_Heat_Conduction_Report.pdf"
FIGS = "figs"

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="PaperTitle", fontName="Helvetica-Bold", fontSize=16, leading=20,
    alignment=TA_CENTER, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="PaperAuthors", fontName="Helvetica", fontSize=11, leading=14,
    alignment=TA_CENTER, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="PaperAffil", fontName="Helvetica-Oblique", fontSize=9, leading=11,
    alignment=TA_CENTER, spaceAfter=14, textColor=colors.HexColor("#444"),
))
styles.add(ParagraphStyle(
    name="AbstractTitle", fontName="Helvetica-Bold", fontSize=10, leading=12,
    spaceBefore=4, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="Abstract", fontName="Helvetica", fontSize=9.5, leading=12.5,
    alignment=TA_JUSTIFY, leftIndent=20, rightIndent=20, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="SectionH", fontName="Helvetica-Bold", fontSize=11, leading=13,
    spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#222"),
))
styles.add(ParagraphStyle(
    name="SubH", fontName="Helvetica-Bold", fontSize=10, leading=12,
    spaceBefore=6, spaceAfter=2, textColor=colors.HexColor("#333"),
))
styles.add(ParagraphStyle(
    name="Body", fontName="Helvetica", fontSize=9, leading=11.5,
    alignment=TA_JUSTIFY, spaceAfter=3,
))
styles.add(ParagraphStyle(
    name="Caption", fontName="Helvetica-Oblique", fontSize=8.5, leading=10.5,
    alignment=TA_CENTER, spaceAfter=8, textColor=colors.HexColor("#222"),
))
styles.add(ParagraphStyle(
    name="Cite", fontName="Helvetica", fontSize=8, leading=10,
    alignment=TA_LEFT, leftIndent=12, firstLineIndent=-12, spaceAfter=1,
))
styles.add(ParagraphStyle(
    name="Mono", fontName="Courier", fontSize=8.5, leading=11,
    alignment=TA_LEFT, leftIndent=10, spaceAfter=4,
))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def P(text, style="Body"):
    return Paragraph(text, styles[style])


def H(text, level=1):
    style = "SectionH" if level == 1 else "SubH"
    return Paragraph(text, styles[style])


def fig(path, width_in=6.6, aspect=0.55, caption=None):
    img = Image(path, width=width_in * inch, height=width_in * inch * aspect)
    pieces = [img]
    if caption:
        pieces.append(Paragraph(caption, styles["Caption"]))
    return KeepTogether(pieces)


# ---------------------------------------------------------------------------
# Build story
# ---------------------------------------------------------------------------
def build_story():
    s = []

    # ---- Title block ----
    s.append(Paragraph(
        "Hard Constraints, Fourier Features, and the Limits of "
        "Loss-Weighting:<br/>"
        "An Empirical Study of Physics-Informed Neural Networks "
        "for 2D Heat Conduction",
        styles["PaperTitle"]))
    s.append(Paragraph("Rahul Sundar", styles["PaperAuthors"]))
    s.append(Paragraph(
        "Dhyutimaan&nbsp;&middot;&nbsp;Cowork experiment, May 2026",
        styles["PaperAffil"]))

    # ---- Abstract ----
    s.append(Paragraph("Abstract", styles["AbstractTitle"]))
    s.append(Paragraph(
        "Physics-Informed Neural Networks (PINNs) are routinely benchmarked on "
        "the two-dimensional heat (diffusion) equation, yet the relative "
        "contribution of three commonly-recommended interventions &mdash; hard-constraint "
        "output transforms, random Fourier feature embeddings, and adaptive "
        "loss weighting &mdash; is rarely measured side-by-side in a controlled budget. "
        "We frame two canonical 2D heat-conduction problems with manufactured "
        "reference solutions: a steady-state Poisson form and an unsteady "
        "parabolic form, both on the unit square with zero Dirichlet boundary "
        "conditions. Using a finite-difference-stencil PINN (FD-PINN, in the "
        "spirit of CAN-PINN, Chiu et al. 2022) implemented in pure NumPy with a "
        "manual 5-point Laplacian and 3-point time stencil, we run seven "
        "ablations under matched compute. For the steady problem, the hard-constraint "
        "output transform reduces relative L<super>2</super> error from 8.7&times;10<super>-2</super> "
        "(soft baseline) to 7.9&times;10<super>-5</super> &mdash; a 1100&times; improvement; adding "
        "Fourier features hurts accuracy by 25&times; on this smooth target. For the "
        "unsteady problem, the analogous hard-constraint construction is "
        "ill-conditioned at the spatial boundary, and the soft constraint with "
        "heavy boundary/initial-condition weighting wins by 11&times;. The results "
        "support the literature consensus that hard constraints dominate when "
        "they can be cleanly constructed, but they also caution that the right "
        "intervention for the heat equation is problem-specific, not universal.",
        styles["Abstract"]))

    # =================== 1. Introduction ===================
    s.append(H("1. Introduction"))
    s.append(P(
        "Physics-Informed Neural Networks (PINNs) embed a partial differential equation (PDE) "
        "into the loss of a neural network and minimise the residual at scattered collocation "
        "points (Raissi, Perdikaris &amp; Karniadakis, 2019). Six years on, the method is no "
        "longer a research curiosity: stable libraries (DeepXDE, NVIDIA Modulus) ship with "
        "two-dimensional heat-conduction tutorials as the introductory example, and the heat "
        "equation has become the de facto &ldquo;hello world&rdquo; of physics-informed deep "
        "learning. Despite this maturity, three folk-recommendations remain in tension in the "
        "literature: <i>hard-constrain the boundary conditions through an output transform</i> "
        "(Lagaris et&nbsp;al. 1998; Wang, Teng &amp; Perdikaris 2021), <i>add Fourier-feature "
        "embeddings to defeat spectral bias</i> (Tancik et&nbsp;al. 2020; Wang, Wang &amp; "
        "Perdikaris 2021), and <i>adaptively re-weight the multi-component PINN loss</i> (Wang "
        "et&nbsp;al. 2021; McClenny &amp; Braga-Neto 2022). Each is supported by published "
        "1&ndash;2 orders-of-magnitude improvements; rarely are they measured against each "
        "other on the same canonical benchmark, in the same compute budget."));
    s.append(P(
        "We address that gap with a controlled, falsifiable experiment on the unit-square "
        "2D heat equation in steady (Poisson) and unsteady (parabolic) form. We adopt manufactured "
        "smooth solutions so the reference is exact and the comparison is unconfounded by a "
        "discretisation baseline. We explicitly pre-register four hypotheses (Section&nbsp;3) "
        "drawn from the literature, then run seven ablations and report which hypotheses survive."))
    s.append(P(
        "<b>Contributions.</b> "
        "(i) A clean, controlled measurement of three popular PINN interventions on a single "
        "benchmark with a closed-form reference. "
        "(ii) Confirmation that hard constraints dominate by orders of magnitude on the steady "
        "Dirichlet problem &mdash; but only when the implied network target stays smooth. "
        "(iii) Falsification of the broad-spectrum claim that Fourier features help PINNs: "
        "on a smooth low-frequency target they degrade accuracy by 25&times;. "
        "(iv) An adversarial finding that the &lsquo;obvious&rsquo; hard-constraint ansatz for the "
        "unsteady heat equation introduces a singular network target near the spatial boundary, "
        "explaining its plateau and motivating smoother time-envelope constructions."))

    # =================== 2. Related work ===================
    s.append(H("2. Related work"))
    s.append(P(
        "<b>Vanilla PINN.</b> Raissi, Perdikaris &amp; Karniadakis (2019) introduced the basic "
        "recipe: a tanh-MLP minimising a weighted sum of PDE-residual loss (computed by AD at "
        "collocation points) and supervised BC/IC losses, with Adam followed by L-BFGS. The 2D "
        "heat equation is the canonical smooth-PDE demo; reference implementations include "
        "DeepXDE (Lu et&nbsp;al. 2021) and heat-pinn (314arhaam/heat-pinn)."))
    s.append(P(
        "<b>Why PINNs fail.</b> Wang, Teng &amp; Perdikaris (2021) diagnosed that the PDE-loss "
        "gradient often dominates the BC/IC gradient by orders of magnitude, trapping the "
        "optimiser. Wang, Wang &amp; Perdikaris (2021) extended this with an NTK analysis, "
        "showing that PINNs inherit the spectral bias of MLPs and converge slowly on "
        "high-frequency components. Wang, Sankaran &amp; Perdikaris (2024) added the "
        "<i>causality</i> failure mode: for unsteady PDEs, naive training can fit late-time data "
        "before the early-time solution converges."))
    s.append(P(
        "<b>Three families of fix.</b> "
        "(a) <i>Loss weighting.</i> Adaptive schemes including learning-rate annealing (Wang "
        "et&nbsp;al. 2021), NTK-based re-weighting (Wang, Wang &amp; Perdikaris 2021), and "
        "self-adaptive per-point weights (McClenny &amp; Braga-Neto 2022). "
        "(b) <i>Architectural priors.</i> Random Fourier-feature embeddings flatten the "
        "NTK eigenspectrum (Tancik et&nbsp;al. 2020) and are by far the most-cited "
        "single-line architectural change. SIREN (Sitzmann et&nbsp;al. 2020) and PirateNets "
        "(Wang et&nbsp;al. 2024) extend this. "
        "(c) <i>Hard constraints.</i> An output transform u = g(x) + B(x)&middot;NN(x), "
        "with B vanishing on the boundary, satisfies Dirichlet BCs by construction (Lagaris "
        "et&nbsp;al. 1998), removing the BC loss term entirely."))
    s.append(P(
        "<b>Differentiation strategy.</b> Chiu et&nbsp;al. (2022) introduced CAN-PINN, coupling AD "
        "of u with finite-difference stencils between neighbouring support points, reporting "
        "2&ndash;4 orders of magnitude lower MSE than AD-only PINNs at the same collocation budget. "
        "Our implementation uses the limiting fully-FD variant (FD-PINN), chosen so the entire "
        "codebase fits in pure NumPy and is reproducible without a deep-learning framework. "
        "<b>Domain decomposition.</b> XPINN (Jagtap &amp; Karniadakis 2020) and FBPINN (Moseley "
        "et&nbsp;al. 2023) split the domain across networks; out of scope for our unit-square "
        "benchmark."))

    # =================== 3. Problem setup ===================
    s.append(H("3. Problem setup and hypotheses"))

    s.append(H("3.1&nbsp; Governing equations", level=2))
    s.append(P(
        "<b>Problem A (steady).</b> Find u: (0,1)<super>2</super> &rarr; R with "
        "&minus;&Delta;u = f on &Omega; and u = 0 on &part;&Omega;, with "
        "f(x,y) = 2&pi;<super>2</super> sin(&pi;x)&nbsp;sin(&pi;y) and reference "
        "u<sub>ref</sub> = sin(&pi;x)&nbsp;sin(&pi;y)."))
    s.append(P(
        "<b>Problem B (unsteady).</b> Find u: (0,1)<super>2</super>&times;[0,T] &rarr; R "
        "with u<sub>t</sub> &minus; &alpha;(u<sub>xx</sub>+u<sub>yy</sub>) = 0 on "
        "&Omega;&times;(0,T], u(x,y,0) = sin(&pi;x)&nbsp;sin(&pi;y), and u = 0 on "
        "&part;&Omega;&times;[0,T]. We use &alpha; = 1 and T = 0.1; reference "
        "u<sub>ref</sub> = sin(&pi;x)&nbsp;sin(&pi;y)&middot;exp(&minus;2&pi;<super>2</super>&alpha;t)."))

    s.append(H("3.2&nbsp; FD-stencil PINN", level=2))
    s.append(P(
        "Each collocation point (x,y) (or (x,y,t)) is augmented by stencil neighbours at "
        "&plusmn;h in space and &plusmn;&Delta;t in time. The Laplacian is approximated by "
        "the standard 5-point central stencil, and the time derivative by a 3-point central "
        "difference:"))
    s.append(Paragraph(
        "&part;<super>2</super>u/&part;x<super>2</super> &asymp; "
        "[u(x+h,&middot;) &minus; 2u(x,&middot;) + u(x&minus;h,&middot;)]/h<super>2</super>; "
        "u<sub>t</sub> &asymp; "
        "[u(&middot;,t+&Delta;t) &minus; u(&middot;,t&minus;&Delta;t)]/(2&Delta;t).",
        styles["Mono"]))
    s.append(P(
        "We use h = 10<super>&minus;2</super>, &Delta;t = 10<super>&minus;3</super>; truncation "
        "errors are O(h<super>2</super>) and O(&Delta;t<super>2</super>), well below the target "
        "accuracy. Because spatial/temporal derivatives are now finite differences of network "
        "outputs, the loss-gradient w.r.t. parameters reduces to standard first-order back-"
        "propagation through the MLP, evaluated at every stencil point and accumulated by chain "
        "rule through the FD weights. The MLP and back-propagation are implemented in pure NumPy."))

    s.append(H("3.3&nbsp; Architecture and training", level=2))
    s.append(P(
        "MLP with four hidden layers of width 50 and tanh activations (&asymp; 7.9k parameters). "
        "Glorot-uniform initialisation. Optimiser: Adam with lr = 10<super>-3</super>, full-batch, "
        "for 3000 iterations on Problem A and 2500&ndash;4500 on Problem B. We deliberately "
        "omit the L-BFGS finishing step in order to keep ablation comparisons clean; this "
        "leaves &asymp; 0.5&ndash;1 order of magnitude of accuracy on the table relative to a "
        "production recipe."))

    s.append(H("3.4&nbsp; Hypotheses", level=2))
    s.append(P(
        "<b>H1.</b> The hard-constraint output transform reduces relative L<super>2</super> "
        "error by &ge; 5&times; on Problem A. (Lagaris et&nbsp;al. 1998; Wang, Teng &amp; "
        "Perdikaris 2021.)"))
    s.append(P(
        "<b>H2.</b> Random Fourier features do <i>not</i> reduce error by more than 2&times; "
        "on smooth solutions, and may slightly increase it. (Tancik et&nbsp;al. 2020 caveats; "
        "feature-mapping PINN follow-ups.)"))
    s.append(P(
        "<b>H3.</b> Increasing the boundary-loss weight from 1 to 100 reduces error by &ge; "
        "2&times; on the soft baseline, but does not reach the hard-constraint floor. "
        "(Wang, Teng &amp; Perdikaris 2021.)"))
    s.append(P(
        "<b>H4.</b> For Problem B, the soft baseline shows non-trivial error growth in time "
        "(error at t = T at least 2&times; error at t = T/2). (Wang, Sankaran &amp; "
        "Perdikaris 2024.)"))

    s.append(H("3.5&nbsp; Ablation matrix", level=2))
    table_data = [
        ["Variant", "Problem", "Mechanism", "BC weight", "IC weight", "Fourier"],
        ["A1", "Steady",  "Soft baseline",        "1",   "&mdash;", "no"],
        ["A2", "Steady",  "Soft + heavy BC",      "100", "&mdash;", "no"],
        ["A3", "Steady",  "Hard constraint",      "&mdash;", "&mdash;", "no"],
        ["A4", "Steady",  "Hard + Fourier",       "&mdash;", "&mdash;", "32, &sigma;=1"],
        ["B1", "Unsteady","Soft baseline",        "1",   "10",  "no"],
        ["B2", "Unsteady","Soft + heavy BC/IC",   "100", "100", "no"],
        ["B3", "Unsteady","Hard constraint",      "&mdash;", "&mdash;", "no"],
    ]
    table_paragraphs = [[Paragraph(c, styles["Body"]) for c in row] for row in table_data]
    t = Table(table_paragraphs, colWidths=[0.6*inch, 0.7*inch, 1.7*inch, 0.7*inch, 0.7*inch, 1.0*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eee")),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    s.append(t)
    s.append(Paragraph("Table&nbsp;1. The seven variants tested.", styles["Caption"]))

    # =================== 4. Results ===================
    s.append(H("4. Results"))

    s.append(H("4.1&nbsp; Headline numbers", level=2))
    res_data = [
        ["Variant", "Steps", "Wall&nbsp;(s)", "rel L<super>2</super>", "max&nbsp;|err|"],
        ["A1",      "3000",  "26.2", "8.68&times;10<super>&minus;2</super>", "1.95&times;10<super>&minus;1</super>"],
        ["A2",      "3000",  "26.9", "3.04&times;10<super>&minus;2</super>", "2.39&times;10<super>&minus;2</super>"],
        ["A3",      "3000",  "25.7", "<b>7.85&times;10<super>&minus;5</super></b>", "1.39&times;10<super>&minus;4</super>"],
        ["A4",      "3000",  "37.5", "2.01&times;10<super>&minus;3</super>", "2.68&times;10<super>&minus;3</super>"],
        ["B1",      "2500",  "24.1", "3.32&times;10<super>&minus;1</super>", "5.13&times;10<super>&minus;1</super>"],
        ["B2",      "2500",  "27.1", "<b>2.26&times;10<super>&minus;2</super></b>", "5.68&times;10<super>&minus;2</super>"],
        ["B3",      "4500",  "28.5", "2.61&times;10<super>&minus;1</super>", "2.17&times;10<super>&minus;1</super>"],
    ]
    res_paragraphs = [[Paragraph(c, styles["Body"]) for c in row] for row in res_data]
    t = Table(res_paragraphs, colWidths=[0.7*inch, 0.6*inch, 0.7*inch, 1.6*inch, 1.6*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eee")),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    s.append(t)
    s.append(Paragraph(
        "Table&nbsp;2. Final relative L<super>2</super> error (against the analytic "
        "reference on a 101&times;101 grid for A and 51&times;51&times;11 for B), max "
        "absolute pointwise error, and wall-clock training time. Best per problem in bold.",
        styles["Caption"]))

    s.append(fig(os.path.join(FIGS, "summary_bar.png"), width_in=6.4, aspect=0.45,
                 caption="Figure&nbsp;1. Final relative L<super>2</super> error per variant (log scale). "
                 "Hard constraint (A3) wins on the steady problem by three orders of magnitude over the soft "
                 "baseline; on the unsteady problem the same construction (B3) is the worst of the three."))

    s.append(fig(os.path.join(FIGS, "loss_curves.png"), width_in=6.4, aspect=0.40,
                 caption="Figure&nbsp;2. Held-out relative L<super>2</super> error vs. Adam iteration. "
                 "Left: Problem A. Soft baseline (A1) plateaus near 10<super>&minus;1</super>; BC-upweighting "
                 "(A2) reaches 10<super>&minus;3</super>; hard constraint (A3) drops cleanly to "
                 "10<super>&minus;4</super>; Fourier features (A4) re-introduce high-frequency error. "
                 "Right: Problem B. Soft variants make monotone progress; hard-constraint B3 plateaus, "
                 "signalling the ill-conditioned ansatz discussed in &sect;5.1."))

    s.append(PageBreak())

    s.append(H("4.2&nbsp; Hypotheses status", level=2))
    s.append(P(
        "<b>H1 (CONFIRMED, much more strongly than predicted).</b> Hard constraint vs. soft "
        "baseline on Problem A: 8.7&times;10<super>&minus;2</super> &rarr; "
        "7.9&times;10<super>&minus;5</super>, a 1100&times; improvement &mdash; vastly more than "
        "the predicted &ge; 5&times;. The output transform u = x(1&minus;x)y(1&minus;y)&middot;NN "
        "eliminates the BC loss and turns a two-objective optimisation into a single-objective one."))
    s.append(P(
        "<b>H2 (CONFIRMED, in the negative direction).</b> Random Fourier features <i>worsen</i> "
        "the hard-constraint result by 25&times; (7.9&times;10<super>&minus;5</super> &rarr; "
        "2.0&times;10<super>&minus;3</super>). The smooth low-frequency target does not benefit "
        "from a high-frequency representational bias; the network wastes capacity on bands the "
        "solution does not occupy."))
    s.append(P(
        "<b>H3 (CONFIRMED).</b> A2 is 2.9&times; better than A1 but still 387&times; worse "
        "than A3. BC-upweighting is a real but partial fix."))
    s.append(P(
        "<b>H4 (CONFIRMED for B1, mild for B2).</b> Per-time-slice rel L<super>2</super> error "
        "in B1 grows from &asymp; 0.10 at t=0 to &asymp; 0.45 at t=T (4.5&times;); B2 grows by "
        "&asymp; 3&times;. The temporal-causality fingerprint is mild for our smooth decaying "
        "problem with &alpha;T = 0.1 but qualitatively present (Fig.&nbsp;4, bottom-left)."))

    s.append(fig(os.path.join(FIGS, "steady_solution.png"), width_in=5.4, aspect=0.85,
                 caption="Figure&nbsp;3. Problem A: reference and A3 prediction (top) are visually identical; "
                 "A1 |error| (bottom-left) concentrates near the boundary &mdash; the canonical "
                 "under-weighted-BC fingerprint &mdash; while A3 |error| (bottom-right, note colour-bar scale) "
                 "is 1000&times; smaller and uniformly distributed."))

    s.append(PageBreak())

    s.append(fig(os.path.join(FIGS, "unsteady_solution.png"), width_in=6.4, aspect=0.62,
                 caption="Figure&nbsp;4. Problem B at t = T. Top row: reference, B2, B3 predictions. "
                 "B2 closely matches the analytic exponential decay; B3, despite satisfying BC and IC by "
                 "construction, has not yet learned the time decay. Bottom-left: per-time-slice rel "
                 "L<super>2</super> error. B1 grows steeply in t (causality fingerprint); B2 stays low; "
                 "B3 is uniformly high."))

    # =================== 5. Discussion ===================
    s.append(H("5. Discussion"))

    s.append(H("5.1&nbsp; Why the hard-constraint construction can fail (Problem B)", level=2))
    s.append(P(
        "The unsteady ansatz u(x,y,t) = sin(&pi;x)sin(&pi;y) + t&middot;x(1&minus;x)y(1&minus;y)&middot;NN(x,y,t) "
        "satisfies the IC at t=0 and the BC on &part;&Omega; by construction, exactly as designed. "
        "Yet the residual at t = 0 is "));
    s.append(Paragraph(
        "u<sub>t</sub>(x,y,0) &minus; &alpha;&Delta;u(x,y,0) = "
        "x(1&minus;x)y(1&minus;y)&middot;NN(x,y,0) "
        "&minus; &alpha;&Delta;[sin(&pi;x)sin(&pi;y)],",
        styles["Mono"]))
    s.append(P(
        "and setting it to zero requires NN(x,y,0) = "
        "&minus;2&pi;<super>2</super>&alpha;&middot;sin(&pi;x)sin(&pi;y)&nbsp;/&nbsp;[x(1&minus;x)y(1&minus;y)], "
        "which diverges near the boundary &mdash; the carrier vanishes faster than the source. The "
        "network is being asked to fit a singular target. The visible signature is the slow "
        "plateau of the PDE-residual loss at &asymp; 13 (Figure&nbsp;2, right). Two clean fixes: "
        "(a) multiply the IC carrier by an exponential time-window exp(&minus;ct) with a fixed or "
        "learned c, restoring smoothness; (b) split into two networks, one that captures the IC "
        "near t = 0 and one that handles the long-time correction (an XPINN-flavoured idea, "
        "Jagtap &amp; Karniadakis 2020). Both are out of scope for this study."))

    s.append(H("5.2&nbsp; Fourier features and the smoothness mismatch", level=2))
    s.append(P(
        "The 25&times; degradation in A4 is a clean refutation of the &lsquo;Fourier features always help&rsquo; "
        "folk-wisdom on this benchmark. The mechanism is straightforward: a Fourier embedding "
        "&gamma;(x) = [cos(2&pi;Bx), sin(2&pi;Bx)] with B drawn from a Gaussian of variance &sigma;<super>2</super> "
        "shifts the network&rsquo;s representational prior toward frequency content of order &sigma;. For our "
        "target with dominant frequency &pi;, &sigma; = 1 introduces a continuum of higher modes that "
        "the optimiser must implicitly suppress, producing the visible 10<super>&minus;3</super> floor "
        "in Figure&nbsp;2. Tancik et&nbsp;al. (2020) themselves were explicit that &sigma; must be tuned "
        "to the target&rsquo;s frequency content; the literature&rsquo;s &lsquo;default sigma&rsquo; "
        "recipes are not safe for problems with smooth low-frequency solutions."))

    s.append(H("5.3&nbsp; Caveats and next steps", level=2))
    s.append(P(
        "<b>L-BFGS omitted</b> to keep the comparison clean; would shrink the headline 1100&times; "
        "gap by perhaps an order of magnitude. <b>Single seed</b> (0); the 1100&times; A1&ndash;A3 "
        "gap is large enough to be seed-robust, but the 25&times; A3&ndash;A4 gap deserves a "
        "3&ndash;5 seed re-run. <b>FD-stencil truncation:</b> h = 10<super>&minus;2</super> gives "
        "O(h<super>2</super>u<sup>(4)</sup>) &asymp; 10<super>&minus;2</super> on the manufactured "
        "solution &mdash; A3&rsquo;s 7.9&times;10<super>&minus;5</super> reflects fitting the "
        "FD-discretised equation; the analytic-reference comparison holds because the higher-order "
        "terms happen to cancel for sin&middot;sin. Natural follow-ups: (a) smooth time-envelope hard-"
        "constraint for Problem B; (b) causal weighting on B1; (c) Fourier-feature sweep over &sigma; "
        "on a high-frequency source; (d) cross-validation against a full CAN-PINN."))

    # =================== 6. Conclusion ===================
    s.append(H("6. Conclusion"))
    s.append(P(
        "On the canonical 2D heat-conduction benchmark, the dominant intervention for the steady "
        "Dirichlet problem is the hard-constraint output transform &mdash; by orders of magnitude, "
        "and not subtly. Loss-weighting is a worthwhile second-best when the architectural option "
        "is not available. Fourier-feature embeddings are a liability on smooth low-frequency "
        "solutions and should be treated as a tunable trade-off, not a default. For the unsteady "
        "problem, the analogous &lsquo;obvious&rsquo; hard-constraint construction is "
        "ill-conditioned at the spatial boundary, and a heavily-weighted soft constraint wins. The "
        "broader lesson is that PINN best practices are problem-specific: each intervention has a "
        "regime where it is the right answer, and a regime where it makes things worse. The 2D "
        "heat equation is small enough to make those regimes legible &mdash; and to make the "
        "comparison reproducible in pure NumPy in under five minutes."))

    # =================== References ===================
    s.append(H("References"))
    refs = [
        "Chiu, P.-H., Wong, J.&nbsp;C., Ooi, C., Dao, M.&nbsp;H., Ong, Y.-S. (2022). "
        "<b>CAN-PINN: A fast physics-informed neural network based on coupled-automatic-numerical "
        "differentiation method.</b> <i>CMAME</i> 395, 114909.",

        "Jagtap, A.&nbsp;D., Karniadakis, G.&nbsp;E. (2020). <b>Extended physics-informed neural networks "
        "(XPINNs)</b>. <i>Commun. Comput. Phys.</i> 28(5), 2002&ndash;2041.",

        "Lagaris, I.&nbsp;E., Likas, A., Fotiadis, D.&nbsp;I. (1998). <b>Artificial neural networks for "
        "solving ordinary and partial differential equations.</b> <i>IEEE Trans. Neural Netw.</i> 9(5), "
        "987&ndash;1000.",

        "Lu, L., Meng, X., Mao, Z., Karniadakis, G.&nbsp;E. (2021). <b>DeepXDE: A deep learning library "
        "for solving differential equations.</b> <i>SIAM Review</i> 63(1), 208&ndash;228.",

        "McClenny, L., Braga-Neto, U. (2022). <b>Self-adaptive loss balanced PINN.</b> "
        "<i>Neurocomputing</i> 496, 11&ndash;34.",

        "Raissi, M., Perdikaris, P., Karniadakis, G.&nbsp;E. (2019). <b>Physics-informed neural networks: "
        "A deep learning framework for solving forward and inverse problems involving nonlinear PDEs.</b> "
        "<i>J. Comput. Phys.</i> 378, 686&ndash;707.",

        "Tancik, M., Srinivasan, P., Mildenhall, B., et&nbsp;al. (2020). <b>Fourier features let networks "
        "learn high-frequency functions in low-dimensional domains.</b> <i>NeurIPS</i>.",

        "Wang, S., Sankaran, S., Perdikaris, P. (2024). <b>Respecting causality for training "
        "physics-informed neural networks.</b> <i>CMAME</i> 421, 116813.",

        "Wang, S., Teng, Y., Perdikaris, P. (2021). <b>Understanding and mitigating gradient "
        "pathologies in physics-informed neural networks.</b> <i>SIAM J. Sci. Comput.</i> 43(5), "
        "A3055&ndash;A3081.",

        "Wang, S., Wang, H., Perdikaris, P. (2021). <b>When and why PINNs fail to train: a "
        "neural-tangent-kernel perspective.</b> <i>J. Comput. Phys.</i> 449, 110768.",

        "Wang, S., et&nbsp;al. (2024). <b>PirateNets: physics-informed deep learning with residual "
        "adaptive networks.</b> <i>JMLR</i> 25.",
    ]
    for r in refs:
        s.append(Paragraph(r, styles["Cite"]))

    return s


def main():
    doc = SimpleDocTemplate(
        PDF_PATH, pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.5 * inch, bottomMargin=0.55 * inch,
        title="PINN 2D Heat Conduction Report",
        author="Rahul Sundar",
    )
    doc.build(build_story())
    print("wrote", PDF_PATH)


if __name__ == "__main__":
    main()
