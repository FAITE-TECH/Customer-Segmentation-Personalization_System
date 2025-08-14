# Import pandas to construct a small sample dataset
import pandas as pd
# Import pathlib to manage output paths
from pathlib import Path

# Define the sample rows provided by the user prompt
SAMPLE_ROWS = [
    # Each tuple represents a row mirroring the Online Retail II schema
    (536365, "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 6, "01/12/2010 08:26", 2.55, 17850, "United Kingdom"),
    (536365, "71053", "WHITE METAL LANTERN", 6, "01/12/2010 08:26", 3.39, 17850, "United Kingdom"),
    (536365, "84406B", "CREAM CUPID HEARTS COAT HANGER", 8, "01/12/2010 08:26", 2.75, 17850, "United Kingdom"),
    (536365, "84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 6, "01/12/2010 08:26", 3.39, 17850, "United Kingdom"),
    (536365, "84029E", "RED WOOLLY HOTTIE WHITE HEART.", 6, "01/12/2010 08:26", 3.39, 17850, "United Kingdom"),
    (536365, "22752", "SET 7 BABUSHKA NESTING BOXES", 2, "01/12/2010 08:26", 7.65, 17850, "United Kingdom"),
    (536365, "21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 6, "01/12/2010 08:26", 4.25, 17850, "United Kingdom"),
    (536366, "22633", "HAND WARMER UNION JACK", 6, "01/12/2010 08:28", 1.85, 17850, "United Kingdom"),
    (536366, "22632", "HAND WARMER RED POLKA DOT", 6, "01/12/2010 08:28", 1.85, 17850, "United Kingdom"),
    (536368, "22960", "JAM MAKING SET WITH JARS", 6, "01/12/2010 08:34", 4.25, 13047, "United Kingdom"),
    (536368, "22913", "RED COAT RACK PARIS FASHION", 3, "01/12/2010 08:34", 4.95, 13047, "United Kingdom"),
    (536368, "22912", "YELLOW COAT RACK PARIS FASHION", 3, "01/12/2010 08:34", 4.95, 13047, "United Kingdom"),
    (536368, "22914", "BLUE COAT RACK PARIS FASHION", 3, "01/12/2010 08:34", 4.95, 13047, "United Kingdom"),
    (536367, "84879", "ASSORTED COLOUR BIRD ORNAMENT", 32, "01/12/2010 08:34", 1.69, 13047, "United Kingdom"),
    (536367, "22745", "POPPY'S PLAYHOUSE BEDROOM", 6, "01/12/2010 08:34", 2.1, 13047, "United Kingdom"),
    (536367, "22748", "POPPY'S PLAYHOUSE KITCHEN", 6, "01/12/2010 08:34", 2.1, 13047, "United Kingdom"),
    (536367, "22749", "FELTCRAFT PRINCESS CHARLOTTE DOLL", 8, "01/12/2010 08:34", 3.75, 13047, "United Kingdom"),
    (536367, "22310", "IVORY KNITTED MUG COSY", 6, "01/12/2010 08:34", 1.65, 13047, "United Kingdom"),
    (536367, "84969", "BOX OF 6 ASSORTED COLOUR TEASPOONS", 6, "01/12/2010 08:34", 4.25, 13047, "United Kingdom"),
    (536367, "22623", "BOX OF VINTAGE JIGSAW BLOCKS", 3, "01/12/2010 08:34", 4.95, 13047, "United Kingdom"),
    (536367, "22622", "BOX OF VINTAGE ALPHABET BLOCKS", 2, "01/12/2010 08:34", 9.95, 13047, "United Kingdom"),
    (536367, "21754", "HOME BUILDING BLOCK WORD", 3, "01/12/2010 08:34", 5.95, 13047, "United Kingdom"),
    (536367, "21755", "LOVE BUILDING BLOCK WORD", 3, "01/12/2010 08:34", 5.95, 13047, "United Kingdom"),
    (536367, "21777", "RECIPE BOX WITH METAL HEART", 4, "01/12/2010 08:34", 7.95, 13047, "United Kingdom"),
    (536367, "48187", "DOORMAT NEW ENGLAND", 4, "01/12/2010 08:34", 7.95, 13047, "United Kingdom"),
    (536369, "21756", "BATH BUILDING BLOCK WORD", 3, "01/12/2010 08:35", 5.95, 13047, "United Kingdom"),
    (536370, "22728", "ALARM CLOCK BAKELIKE PINK", 24, "01/12/2010 08:45", 3.75, 12583, "France"),
    (536370, "22727", "ALARM CLOCK BAKELIKE RED", 24, "01/12/2010 08:45", 3.75, 12583, "France"),
    (536370, "22726", "ALARM CLOCK BAKELIKE GREEN", 12, "01/12/2010 08:45", 3.75, 12583, "France"),
    (536370, "21724", "PANDA AND BUNNIES STICKER SHEET", 12, "01/12/2010 08:45", 0.85, 12583, "France"),
]

# Convert the tuples into a pandas DataFrame with proper column names
df = pd.DataFrame(
    SAMPLE_ROWS,
    columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"],
)

# Ensure the output directory exists
out_dir = Path(__file__).resolve().parent
# Define CSV output path to be read by the training script if xlsx is absent
csv_path = out_dir / "online_retail_II_sample.csv"
# Parse the InvoiceDate to a datetime string formatted consistently
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True)
# Write the CSV to disk including the parsed dates in ISO format
df.to_csv(csv_path, index=False)
# Print a confirmation message for the developer running the script
print(f"Wrote sample CSV to {csv_path}")
