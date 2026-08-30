from src.metadata import load_queries, extract_choices
from src.domain_router import DomainRouter


df, _ = load_queries("dev")

router = DomainRouter()

print("\nDOMAIN ROUTING TEST")
print("=" * 70)

for _, row in df.iterrows():

    choices = extract_choices(row)

    domain = router.route(
        row["question"],
        choices,
    )

    print(
        f"Query {row['query_id']:>2} "
        f"-> {domain}"
    )
    
