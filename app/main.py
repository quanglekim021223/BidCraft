"""Main entry point for ProposalCraft CLI application"""

from app.handlers.proposal_handler import ProposalHandler


def main():
    """Main function - CLI entry point"""
    print("=" * 50)
    print("🚀 ProposalCraft - Proposal Generator")
    print("=" * 50)
    print()
    
    handler = ProposalHandler()
    output_file = handler.generate_proposal()
    
    if output_file:
        print("=" * 50)
        print(f"✅ COMPLETED!")
        print(f"📁 Output file: {output_file}")
        print("=" * 50)
    else:
        print("=" * 50)
        print("❌ Failed to generate proposal")
        print("=" * 50)


if __name__ == "__main__":
    main()

