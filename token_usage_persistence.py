"""Token usage persistence utilities"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from token_tracking import TokenUsage

logger = logging.getLogger(__name__)


class TokenUsagePersistence:
    """Handles persistence of token usage data"""

    def __init__(self, storage_path: str = "token_usage.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized TokenUsagePersistence with storage: {storage_path}")

    def save_usage(self, token_usage: TokenUsage, session_id: str = None) -> bool:
        """Save token usage to persistent storage"""
        try:
            # Load existing data
            existing_data = self.load_all_usage()

            # Add or update session data
            session_data = {
                "session_id": session_id or f"session_{token_usage.input_tokens + token_usage.output_tokens}",
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
                "total_tokens": token_usage.input_tokens + token_usage.output_tokens,
                "estimated_cost": token_usage.get_estimated_cost(),
                "timestamp": self._get_current_timestamp()
            }

            # Update session in existing data
            if "sessions" not in existing_data:
                existing_data["sessions"] = {}
            existing_data["sessions"][session_id] = session_data

            # Keep only last 100 sessions to prevent file from growing too large
            if len(existing_data["sessions"]) > 100:
                # Convert to list, keep last 100, convert back to dict
                session_items = list(existing_data["sessions"].items())[-100:]
                existing_data["sessions"] = dict(session_items)

            # Save to file
            with open(self.storage_path, 'w') as f:
                json.dump(existing_data, f, indent=2)

            logger.info(f"Saved token usage for session {session_id or 'latest'}")
            return True

        except Exception as e:
            logger.error(f"Failed to save token usage: {str(e)}")
            return False

    def load_usage(self, session_id: str = None) -> Optional[TokenUsage]:
        """Load token usage for specific session or latest"""
        try:
            data = self.load_all_usage()

            if session_id and session_id in data.get("sessions", {}):
                session_data = data["sessions"][session_id]
                return TokenUsage(
                    input_tokens=session_data["input_tokens"],
                    output_tokens=session_data["output_tokens"]
                )
            elif "sessions" in data and data["sessions"]:
                # Return latest session
                latest_session = data["sessions"][-1]
                return TokenUsage(
                    input_tokens=latest_session["input_tokens"],
                    output_tokens=latest_session["output_tokens"]
                )
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to load token usage: {str(e)}")
            return None

    def load_all_usage(self) -> Dict[str, Any]:
        """Load all token usage data from storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            else:
                return {"sessions": {}}
        except Exception as e:
            logger.error(f"Failed to load token usage data: {str(e)}")
            return {"sessions": {}}

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of all token usage"""
        data = self.load_all_usage()

        sessions_dict = data.get("sessions", {})
        sessions_list = list(sessions_dict.values())

        total_input = sum(session["input_tokens"] for session in sessions_list)
        total_output = sum(session["output_tokens"] for session in sessions_list)
        total_cost = sum(session.get("estimated_cost", 0) for session in sessions_list)

        return {
            "total_sessions": len(sessions_list),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_estimated_cost": total_cost,
            "recent_sessions": sessions_list[-5:]  # Last 5 sessions
        }

    def clear_old_sessions(self, keep_last: int = 50) -> bool:
        """Clear old sessions, keeping only the most recent ones"""
        try:
            data = self.load_all_usage()
            sessions_dict = data.get("sessions", {})

            if len(sessions_dict) > keep_last:
                # Sort by timestamp and keep last N
                sorted_sessions = sorted(sessions_dict.items(),
                                       key=lambda x: x[1].get('timestamp', ''),
                                       reverse=True)
                kept_sessions = dict(sorted_sessions[:keep_last])
                data["sessions"] = kept_sessions

                with open(self.storage_path, 'w') as f:
                    json.dump(data, f, indent=2)

                logger.info(f"Cleared old sessions, keeping last {keep_last}")
                return True

        except Exception as e:
            logger.error(f"Failed to clear old sessions: {str(e)}")
            return False

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()
