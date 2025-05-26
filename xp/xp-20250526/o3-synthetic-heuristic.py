import pandas as pd

# Load the original CSV
df = pd.read_csv('ipcc_statements_trimmed.csv')

def assign_confidence(stmt: str) -> str:
    """Assign IPCC-style confidence level based on common qualifiers and language cues."""
    s = str(stmt).lower()
    if any(phrase in s for phrase in ('very high confidence', 'virtually certain', 'extremely likely')):
        return 'very high'
    if any(phrase in s for phrase in ('high confidence', 'very likely')):
        return 'high'
    if 'medium confidence' in s:
        return 'medium'
    if any(phrase in s for phrase in ('low confidence', 'unlikely')):
        return 'low'
    # Qualifiers suggesting moderate evidence / higher uncertainty
    if 'likely' in s:
        return 'medium'
    if any(w in s for w in ['projected', 'project', 'model', 'simulation', 'suggest', 'could', 'may', 'depends', 'uncertain', 'future']):
        return 'medium'
    # Default assumption for well‑established statements
    return 'high'

# Apply the heuristic
df['confidence'] = df['statement'].apply(assign_confidence)

# Save the augmented CSV
output_path = 'ipcc_statements_with_confidence_o3.csv'
df.to_csv(output_path, index=False)

# # Show the first few lines to the user
# import ace_tools as tools; tools.display_dataframe_to_user("IPCC Statements with Confidence", df.head())


