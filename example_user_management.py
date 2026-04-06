"""Example usage of receipt processing with user management"""

import os
from uuid import uuid4
from dotenv import load_dotenv

from image_processing import VisionProcessor
from ai_parsing import ReceiptParser
from receipt_processor import ReceiptProcessor
from database_models import DatabaseManager

# Load environment variables
load_dotenv()

def demo_single_user_mode():
    """Demonstrate single-user mode (default behavior)"""
    print("🏠 Single-User Mode Demo")
    print("=" * 40)
    
    # Initialize database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///receipts.db")
    db_manager = DatabaseManager(DATABASE_URL)
    db_manager.create_tables()
    
    # Initialize processor (automatically creates default user)
    processor = ReceiptProcessor(db_manager=db_manager)
    
    # Get user context
    user_context = processor.get_user_context()
    print(f"Current user: {user_context['email']}")
    print(f"User ID: {user_context['user_id']}")
    print(f"Multi-user mode: {user_context['is_multi_user']}")
    
    # Process a receipt
    image_path = "path/to/receipt.jpg"
    print(f"\n📝 Processing receipt for {user_context['email']}...")
    # result = processor.process_directly(image_path)
    
    # Get user's receipts
    receipts_result = processor.get_user_receipts(limit=5)
    if receipts_result.success:
        print(f"📋 User has {len(receipts_result.data['receipts'])} receipts")
    
    db_manager.close()

def demo_multi_user_mode():
    """Demonstrate multi-user mode capabilities"""
    print("\n👥 Multi-User Mode Demo")
    print("=" * 40)
    
    # Set multi-user mode
    os.environ["MULTI_USER_MODE"] = "true"
    
    # Initialize database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///receipts.db")
    db_manager = DatabaseManager(DATABASE_URL)
    db_manager.create_tables()
    
    # Initialize processor
    processor = ReceiptProcessor(db_manager=db_manager)
    
    # Create and switch between users
    users = ["alice@example.com", "bob@example.com", "charlie@example.com"]
    
    for email in users:
        print(f"\n👤 Switching to user: {email}")
        user = processor.switch_user(email)
        
        # Show user context
        context = processor.get_user_context()
        print(f"   User ID: {context['user_id']}")
        print(f"   Email: {context['email']}")
        
        # Simulate processing receipts for each user
        print(f"   📝 Processing receipts for {email}...")
        # result = processor.process_directly("receipt.jpg")
        
        # Get user's receipts
        receipts_result = processor.get_user_receipts(limit=3)
        if receipts_result.success:
            print(f"   📋 {email} has {len(receipts_result.data['receipts'])} receipts")
    
    # Switch back to first user to demonstrate data isolation
    print(f"\n🔄 Switching back to Alice...")
    processor.switch_user("alice@example.com")
    alice_receipts = processor.get_user_receipts()
    print(f"📋 Alice's receipts: {len(alice_receipts.data['receipts'])} (isolated from Bob/Charlie)")
    
    db_manager.close()

def demo_backward_compatibility():
    """Demonstrate backward compatibility (no database)"""
    print("\n🔄 Backward Compatibility Demo")
    print("=" * 40)
    
    # Initialize processor without database
    processor = ReceiptProcessor()  # No db_manager = no persistence
    
    # User management not available without database
    user_context = processor.get_user_context()
    print(f"User context: {user_context}")
    
    # Process receipt (in-memory only)
    image_path = "path/to/receipt.jpg"
    print(f"📝 Processing receipt without persistence...")
    # result = processor.process_directly(image_path)
    
    # Receipt management not available without database
    receipts_result = processor.get_user_receipts()
    print(f"📋 Receipts result: {receipts.error}")
    
    print("✅ Backward compatibility maintained")

def demo_user_switching_workflow():
    """Demonstrate practical user switching workflow"""
    print("\n🔄 User Switching Workflow Demo")
    print("=" * 45)
    
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///receipts.db")
    db_manager = DatabaseManager(DATABASE_URL)
    db_manager.create_tables()
    
    processor = ReceiptProcessor(db_manager=db_manager)
    
    # Simulate a family sharing the app
    family_members = ["dad@family.com", "mom@family.com", "kid@family.com"]
    
    print("🏠 Family Receipt Processing")
    
    for member_email in family_members:
        # Switch to family member
        user = processor.switch_user(member_email)
        print(f"\n👤 {member_email}'s receipts:")
        
        # Get their receipts
        receipts_result = processor.get_user_receipts(limit=2)
        if receipts_result.success:
            for receipt in receipts_result.data['receipts']:
                print(f"   🧾 {receipt['merchant_name']}: ${receipt['total_amount']} ({receipt['receipt_date']})")
        else:
            print(f"   📭 No receipts yet")
    
    print(f"\n💡 Benefits:")
    print(f"   • Each family member has isolated data")
    print(f"   • Easy switching between users")
    print(f"   • Shared app, separate receipts")
    
    db_manager.close()

if __name__ == "__main__":
    print("🧪 User Management Functionality Demo")
    print("=" * 50)
    
    # Demo 1: Single-user mode (default)
    demo_single_user_mode()
    
    # Demo 2: Multi-user mode
    demo_multi_user_mode()
    
    # Demo 3: Backward compatibility
    demo_backward_compatibility()
    
    # Demo 4: Practical workflow
    demo_user_switching_workflow()
    
    print("\n✨ Summary:")
    print("   • Default single-user mode for simplicity")
    print("   • Easy multi-user expansion when needed")
    print("   • Full backward compatibility maintained")
    print("   • User data isolation and security")
    print("   • Simple API for user management")
