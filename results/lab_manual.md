# The Physicist's Laboratory Manual (v1.0)

This manual is your primary reference for building stable, valid MuJoCo simulations using the `LaboratoryBuilder`.

## I. Structural Schema Rules (MJCF)

### 1. Parentage Constraints
- **Bodies**: Must be children of the `<worldbody>`.
- **Children**: `<joint>`, `<geom>`, and `<site>` elements **MUST** be children of a `<body>`. They will fail if added to the `world` directly.
- **Tendons**: `<fixed>` tendons must connect **EXACTLY TWO** `<site>` elements. These sites must belong to different bodies for relative motion.

### 2. Degrees of Freedom (DOF)
- A body with no joints is **welded** to its parent (or the world).
- To make a body move freely (like a falling ball), use `dynamic=True` which adds a `<freejoint>`.
- **Constraint Error**: "either both body1 and anchor must be defined, or both site1 and site2..."
    - *Fix*: Ensure your `add_tendon` call provides two valid site names that have already been added to their respective bodies.

## II. Physical Stability Matrix

| Material | Density | Default Friction | SolRef / SolImp | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **steel** | 7800 | 0.5 | 0.02 1.0 | Standard rigid structures. |
| **rubber** | 1100 | 1.0 | 0.01 0.8 | High-bounce, high-friction. |
| **ice** | 900 | 0.05 | 0.02 1.0 | Low-friction sliding. |

### Stability Tips
- **Mass Ratios**: Avoid connecting a 1,000kg body to a 0.01kg body with a single joint; this often causes "Divergence" (NaN).
- **High-Force Damping**: For heavy robotic arms, increase `damping` in `add_joint` to `100` or higher to prevent jitter.
- **Integration**: We use **RK4** at **500Hz**. This is stable for most link-ratios, but fast rotations still require careful damping.

## III. Troubleshooting [Builder Errors]

- **"cannot serialize [FLOAT]"**: Ensure all parameters are passed as strings or integers.
- **"Body 'X' not found"**: You tried to add a site/joint/geom before adding the body. Always call `add_box`/`add_sphere` first.
- **"Schema violation: unrecognized element"**: You placed an element in the wrong hierarchy (e.g., `<site>` as a child of `<worldbody>`).
- **"Tendon Error"**: Check that both sites exist and are named correctly.

## IV. The Research Workflow
1. **Derive**: Use `sympy_derive` for symbolic verification.
2. **Consult**: If building a complex mechanism, use `read_manual()` to check for parentage rules.
3. **Build**: Use `construct_laboratory` using the verified patterns.
4. **Audit**: Use `get_mass_properties` to confirm build integrity.
