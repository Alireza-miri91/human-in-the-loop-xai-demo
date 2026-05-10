from flask import Flask, session, redirect, url_for, request, jsonify
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
import random
import os
import time
import json
import threading
import math
import secrets
from datetime import timedelta

# Import the two group apps
try:
    from .control_app import get_app as get_control_app
    from .treatment_app import get_app as get_treatment_app
except ImportError:  # Allows running this file directly during local development.
    from control_app import get_app as get_control_app
    from treatment_app import get_app as get_treatment_app

# Public demo group names. They are intentionally descriptive instead of
# obfuscated because this repository is not running a live blinded study.
CONTROL_GROUP = "control"
TREATMENT_GROUP = "treatment"

# BLOCKED RANDOMIZATION CONFIGURATION
BLOCK_SIZE = 4  # Balance every 4 participants (can be 2, 4, 6, 8, etc.)
ENABLE_BLOCKED_RANDOMIZATION = True  # Set to False to use old biased coin method

SECRET_KEY = os.environ.get("ROUTER_SECRET_KEY") or secrets.token_hex(32)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DATA_DIR = os.environ.get(
    "HIL_XAI_RUNTIME_DATA_DIR",
    os.path.join(PROJECT_ROOT, "runtime_data"),
)
ENABLE_ADMIN_ROUTES = os.environ.get("HIL_XAI_ENABLE_ADMIN_ROUTES", "").lower() in {
    "1",
    "true",
    "yes",
}
os.makedirs(RUNTIME_DATA_DIR, exist_ok=True)

router = Flask(__name__)
router.secret_key = SECRET_KEY

# Make the router's session cookie distinct so sub-apps can't clobber it
router.config.update(
    SESSION_COOKIE_NAME='router_sid',        # Different from default 'session'
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,             # Set True when you serve over HTTPS
    PERMANENT_SESSION_LIFETIME=timedelta(days=365),
    SESSION_REFRESH_EACH_REQUEST=False,
)

# Local runtime files are ignored by git and never include IPs or user agents.
ASSIGN_LOG = os.path.join(RUNTIME_DATA_DIR, "assignments.jsonl")
ENGAGEMENT_LOG = os.path.join(RUNTIME_DATA_DIR, "engagement.jsonl")
NEW_USERS_LOG = os.path.join(RUNTIME_DATA_DIR, "new_users.jsonl")
INTERACTIONS_LOG = os.path.join(RUNTIME_DATA_DIR, "interactions.jsonl")

_LOCK = threading.Lock()

def _load_counts():
    """Load current assignment counts from the log file"""
    control_count = treatment_count = 0
    if os.path.exists(ASSIGN_LOG):
        with open(ASSIGN_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    g = json.loads(line).get("group")
                    if g == CONTROL_GROUP: 
                        control_count += 1
                    elif g == TREATMENT_GROUP: 
                        treatment_count += 1
                except Exception:
                    continue
    return control_count, treatment_count

def _load_new_users_counts():
    """Load counts of genuinely new users (first-time visitors)"""
    control_new_users = treatment_new_users = 0
    if os.path.exists(NEW_USERS_LOG):
        with open(NEW_USERS_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    g = json.loads(line).get("group")
                    if g == CONTROL_GROUP: 
                        control_new_users += 1
                    elif g == TREATMENT_GROUP: 
                        treatment_new_users += 1
                except Exception:
                    continue
    return control_new_users, treatment_new_users

def _is_new_user(pid):
    """Check if this is a genuinely new user (first-time visitor)"""
    if os.path.exists(NEW_USERS_LOG):
        with open(NEW_USERS_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("pid") == pid:
                        return False  # User already exists
                except Exception:
                    continue
    return True  # New user

def _log_new_user(pid, group):
    """Log a genuinely new user (first-time visitor)"""
    rec = {
        "ts": time.time(),
        "pid": pid,
        "group": group,
        "note": "First-time demo visitor",
    }
    with _LOCK:
        with open(NEW_USERS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

def _resolve_pid():
    """Resolve PID from session or fallback cookie, creating new if needed"""
    pid = session.get("pid") or request.cookies.get("router_pid")
    if not pid:
        pid = secrets.token_urlsafe(16)
        print(f"🔍 DEBUG: Generated new PID: {pid}")
    else:
        print(f"🔍 DEBUG: Using PID (session/cookie): {pid}")
    session["pid"] = pid
    return pid

def _load_engagement_counts():
    """Load engagement metrics for each group"""
    control_engagement = {
        'page_views': 0,
        'sessions': 0,
        'total_duration': 0,
        'interactions': 0,
        'unique_users': set()
    }
    treatment_engagement = {
        'page_views': 0,
        'sessions': 0,
        'total_duration': 0,
        'interactions': 0,
        'unique_users': set()
    }
    
    if os.path.exists(ENGAGEMENT_LOG):
        with open(ENGAGEMENT_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    group = rec.get("group")
                    if group == CONTROL_GROUP:
                        control_engagement['page_views'] += rec.get('page_views', 0)
                        control_engagement['sessions'] += rec.get('sessions', 0)
                        control_engagement['total_duration'] += rec.get('duration', 0)
                        control_engagement['interactions'] += rec.get('interactions', 0)
                        if 'pid' in rec:
                            control_engagement['unique_users'].add(rec['pid'])
                    elif group == TREATMENT_GROUP:
                        treatment_engagement['page_views'] += rec.get('page_views', 0)
                        treatment_engagement['sessions'] += rec.get('sessions', 0)
                        treatment_engagement['total_duration'] += rec.get('duration', 0)
                        treatment_engagement['interactions'] += rec.get('interactions', 0)
                        if 'pid' in rec:
                            treatment_engagement['unique_users'].add(rec['pid'])
                except Exception:
                    continue
    
    # Convert sets to counts
    control_engagement['unique_users'] = len(control_engagement['unique_users'])
    treatment_engagement['unique_users'] = len(treatment_engagement['unique_users'])
    
    return control_engagement, treatment_engagement

def _log_engagement(pid, group, page_views=1, duration=0, interactions=0):
    """Log user engagement metrics"""
    rec = {
        "ts": time.time(),
        "pid": pid,
        "group": group,
        "page_views": page_views,
        "duration": duration,
        "interactions": interactions,
    }
    with _LOCK:
        with open(ENGAGEMENT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

def _get_existing_assignment(pid):
    """Check if a participant ID already has a group assignment in the log"""
    if not pid:
        print(f"   🔍 DEBUG: No PID provided to _get_existing_assignment")
        return None
        
    if os.path.exists(ASSIGN_LOG):
        try:
            line_count = 0
            with open(ASSIGN_LOG, "r", encoding="utf-8") as f:
                for line_count, line in enumerate(f, 1):
                    try:
                        rec = json.loads(line)
                        if rec.get("pid") == pid:
                            group = rec.get("group")
                            print(f"   🔍 Found existing assignment: PID {pid} -> Group {group} (line {line_count})")
                            return group
                    except Exception as e:
                        print(f"   ⚠️  Error parsing line {line_count}: {e}")
                        continue
            print(f"   🔍 No existing assignment found for PID {pid} in {line_count} lines")
        except Exception as e:
            print(f"   ❌ Error reading assignment log: {e}")
    else:
        print(f"   🔍 Assignment log file does not exist yet")
    
    return None

def _biased_coin_choice():
    """
    Efron biased-coin randomization:
    - If perfectly tied, p(control)=0.5
    - If control is behind, p(control)=2/3; if ahead, p(control)=1/3
    This keeps groups well balanced without being predictable.
    """
    print(f"   🎲 Running biased coin randomization...")
    a, b = _load_counts()
    print(f"   📊 Current counts - Control: {a}, Treatment: {b}")
    
    if a == b:
        pControl = 0.5
        print(f"   ⚖️  Groups tied - p(Control) = 0.5")
    elif a < b:
        pControl = 2/3
        print(f"   ⚖️  Control behind - p(Control) = 2/3")
    else:
        pControl = 1/3
        print(f"   ⚖️  Control ahead - p(Control) = 1/3")
    
    result = CONTROL_GROUP if random.random() < pControl else TREATMENT_GROUP
    print(f"   🎯 Biased coin result: {result}")
    return result

def _blocked_randomization():
    """
    Permuted block randomization for small sample sizes.
    Ensures perfect balance at regular intervals.
    
    Block size of 4 means:
    - Every 4 participants, groups will be perfectly balanced
    - Maximum imbalance is 2 participants
    - Within each block, order is randomized for unpredictability
    """
    print(f"   🎲 Running blocked randomization (block size: {BLOCK_SIZE})...")
    a, b = _load_counts()
    total = a + b
    
    # Calculate position within current block
    position_in_block = total % BLOCK_SIZE
    
    if position_in_block == 0:
        # Start of new block - randomize the order for this block
        # Create all possible balanced block patterns
        if BLOCK_SIZE == 2:
            block_patterns = ['AB', 'BA']
        elif BLOCK_SIZE == 4:
            block_patterns = ['AABB', 'ABAB', 'ABBA', 'BAAB', 'BABA', 'BBAA']
        elif BLOCK_SIZE == 6:
            block_patterns = ['AAABBB', 'AABABB', 'AABBAB', 'AABBBA', 'ABAABB', 'ABABAB', 
                            'ABABBA', 'ABBAAB', 'ABBABA', 'ABBBAA', 'BAAABB', 'BAABAB',
                            'BAABBA', 'BABAAB', 'BABABA', 'BABBAA', 'BBAAAB', 'BBAABA',
                            'BBABAA', 'BBBAAA']
        else:
            # For other block sizes, generate balanced patterns dynamically
            control_count = BLOCK_SIZE // 2
            treatment_count = BLOCK_SIZE // 2
            # This is a simplified approach - in practice you'd want more sophisticated pattern generation
            block_patterns = ['A' * control_count + 'B' * treatment_count]
        
        # Randomly select a block pattern
        selected_pattern = random.choice(block_patterns)
        print(f"   🧱 New block starting - selected pattern: {selected_pattern}")
        
        # Store the block pattern in session for this block
        session['current_block_pattern'] = selected_pattern
        session['block_start_position'] = total
        
        # Return first assignment from the pattern
        first_assignment = selected_pattern[0]
        result = CONTROL_GROUP if first_assignment == 'A' else TREATMENT_GROUP
        print(f"   🎯 Block pattern {selected_pattern} - first assignment: {result}")
        return result
    else:
        # Continue current block - use stored pattern
        current_pattern = session.get('current_block_pattern')
        block_start = session.get('block_start_position', 0)
        
        if current_pattern and (total - block_start) < len(current_pattern):
            # Use the stored pattern
            pattern_index = total - block_start
            assignment = current_pattern[pattern_index]
            result = CONTROL_GROUP if assignment == 'A' else TREATMENT_GROUP
            print(f"   🧱 Continuing block pattern {current_pattern} - position {pattern_index+1}: {result}")
            return result
        else:
            # Fallback: use simple balancing within current block
            if a < b:
                result = CONTROL_GROUP
                print(f"   ⚖️  Control behind in current block - forcing Control assignment")
            else:
                result = TREATMENT_GROUP
                print(f"   ⚖️  Treatment behind in current block - forcing Treatment assignment")
            return result

def _log_assignment(pid, group):
    """Log assignment details to JSONL file"""
    rec = {
        "ts": time.time(),
        "pid": pid,
        "group": group,
    }
    with _LOCK:
        with open(ASSIGN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

@router.route('/')
def assign_group():
    """Main entry point - assigns users to groups and redirects"""
    # Make the session sticky across browser restarts
    session.permanent = True
    
    print(f"\n🔍 DEBUG: Starting group assignment...")
    
    # Use the new PID resolution that can fall back to cookies
    pid = _resolve_pid()
    print(f"🔍 DEBUG: Session data: {dict(session)}")
    
    # CRITICAL: ALWAYS check the log file FIRST for existing assignments
    # This is the source of truth - once assigned, never change
    print(f"🔍 DEBUG: Checking log file for existing assignment...")
    existing_group = _get_existing_assignment(pid)
    print(f"🔍 DEBUG: Existing group from log: {existing_group}")
    
    if existing_group:
        # User already has a group assignment - RESTORE and NEVER CHANGE
        group = existing_group
        session["group"] = group  # Update session with the existing group
        print(f"🔒 RESTORED existing group {group} for PID {pid} from log")
        print(f"   ⚠️  This user will ALWAYS stay in group {group}")
        
        # FINAL SAFETY CHECK: Verify this user is NEVER reassigned
        if existing_group not in [CONTROL_GROUP, TREATMENT_GROUP]:
            print(f"   ❌ ERROR: Invalid group {existing_group} found in log!")
            print(f"   🔧 Resetting to safe default...")
            group = CONTROL_GROUP  # Safe fallback
            _log_assignment(pid, group)  # Overwrite invalid assignment
        else:
            print(f"   ✅ Group {existing_group} is valid and will be preserved")
    else:
        print(f"🔍 DEBUG: No existing assignment found, checking session...")
        # User has NO existing assignment - check session
        session_group = session.get("group")
        print(f"🔍 DEBUG: Session group: {session_group}")
        
        if session_group and session_group in [CONTROL_GROUP, TREATMENT_GROUP]:
            # User has group in session but not in log - this shouldn't happen normally
            # Log it to prevent future issues
            group = session_group
            _log_assignment(pid, group)
            print(f"📝 Logged existing session group {group} for PID {pid}")
            print(f"   ⚠️  This was an edge case - user now permanently assigned to {group}")
        else:
            print(f"🔍 DEBUG: No session group, assigning new group...")
            # Completely new user - assign new group using blocked randomization or biased coin
            if ENABLE_BLOCKED_RANDOMIZATION:
                group = _blocked_randomization()
            else:
                group = _biased_coin_choice()
            session["group"] = group
            _log_assignment(pid, group)
            print(f"🎯 Assigned NEW group {group} to PID {pid}")
            print(f"   ✅ This user is now permanently assigned to {group}")
            
            # Check if this is a genuinely new user (first-time visitor)
            if _is_new_user(pid):
                _log_new_user(pid, group)
                print(f"🎉 NEW USER DETECTED! PID {pid} is a first-time visitor")
            else:
                print(f"🔄 Returning user PID {pid} - not counting as new")
    
    # FINAL VERIFICATION: Ensure group is valid before redirect
    if group not in [CONTROL_GROUP, TREATMENT_GROUP]:
        print(f"   ❌ CRITICAL ERROR: Invalid group {group} for PID {pid}")
        print(f"   🔧 Using safe fallback group {CONTROL_GROUP}")
        group = CONTROL_GROUP
        session["group"] = CONTROL_GROUP
    
    print(f"   🎯 FINAL GROUP ASSIGNMENT: PID {pid} -> Group {group}")
    print(f"🔍 DEBUG: Final session data: {dict(session)}")
    
    # Redirect to appropriate group with sticky cookies
    target = f'/{CONTROL_GROUP}/' if group == CONTROL_GROUP else f'/{TREATMENT_GROUP}/'
    resp = redirect(target)
    
    # Harden persistence with explicit cookies (in addition to Flask session)
    one_year = 60 * 60 * 24 * 365
    resp.set_cookie(
        'router_pid', pid, max_age=one_year, httponly=True, samesite='Lax',
        secure=router.config['SESSION_COOKIE_SECURE'], path='/'
    )
    resp.set_cookie(
        'router_group', group, max_age=one_year, httponly=True, samesite='Lax',
        secure=router.config['SESSION_COOKIE_SECURE'], path='/'
    )
    
    return resp

@router.route('/group')
def group_redirect():
    group = session.get('group') or request.cookies.get('router_group')
    if group == CONTROL_GROUP:
        return redirect(f'/{CONTROL_GROUP}/')
    elif group == TREATMENT_GROUP:
        return redirect(f'/{TREATMENT_GROUP}/')
    else:
        return redirect('/')

@router.route('/metrics')
def metrics():
    """Endpoint to check assignment balance and statistics"""
    control_count, treatment_count = _load_counts()
    control_engagement, treatment_engagement = _load_engagement_counts()
    n = control_count + treatment_count
    
    # Calculate z-score and p-value for deviation from 50/50
    if n == 0:
        pval = 1.0
        z = 0.0
    else:
        z = (control_count - 0.5 * n) / math.sqrt(n * 0.25)
        # Normal CDF via erf for two-sided p-value
        pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    
    # Calculate balance percentage
    balance_control = (control_count / n * 100) if n > 0 else 0
    balance_treatment = (treatment_count / n * 100) if n > 0 else 0
    
    # Calculate engagement metrics
    avg_control_duration = (control_engagement['total_duration'] / control_engagement['sessions']) if control_engagement['sessions'] > 0 else 0
    avg_treatment_duration = (treatment_engagement['total_duration'] / treatment_engagement['sessions']) if treatment_engagement['sessions'] > 0 else 0
    
    avg_control_interactions = (control_engagement['interactions'] / control_engagement['unique_users']) if control_engagement['unique_users'] > 0 else 0
    avg_treatment_interactions = (treatment_engagement['interactions'] / treatment_engagement['unique_users']) if treatment_engagement['unique_users'] > 0 else 0
    
    return jsonify({
        "control_group": CONTROL_GROUP,
        "treatment_group": TREATMENT_GROUP,
        "control_count": control_count,
        "treatment_count": treatment_count,
        "N": n,
        "balance_control_percent": round(balance_control, 2),
        "balance_treatment_percent": round(balance_treatment, 2),
        "difference": abs(control_count - treatment_count),
        "z_score": round(z, 3),
        "p_value_two_sided": round(pval, 4),
        "is_balanced": pval > 0.05,  # 95% confidence level
        "note": f"Control group: {CONTROL_GROUP}, Treatment group: {TREATMENT_GROUP}",
        "engagement": {
            "control": {
                "page_views": control_engagement['page_views'],
                "sessions": control_engagement['sessions'],
                "total_duration": control_engagement['total_duration'],
                "avg_duration": round(avg_control_duration, 2),
                "interactions": control_engagement['interactions'],
                "avg_interactions_per_user": round(avg_control_interactions, 2),
                "unique_users": control_engagement['unique_users']
            },
            "treatment": {
                "page_views": treatment_engagement['page_views'],
                "sessions": treatment_engagement['sessions'],
                "total_duration": treatment_engagement['total_duration'],
                "avg_duration": round(avg_treatment_duration, 2),
                "interactions": treatment_engagement['interactions'],
                "avg_interactions_per_user": round(avg_treatment_interactions, 2),
                "unique_users": treatment_engagement['unique_users']
            }
        }
    })

@router.route('/new-users')
def new_users_metrics():
    """Endpoint specifically for new users (first-time visitors) metrics"""
    control_new_users, treatment_new_users = _load_new_users_counts()
    total_new_users = control_new_users + treatment_new_users
    
    # Calculate balance for new users only
    if total_new_users == 0:
        pval = 1.0
        z = 0.0
    else:
        z = (control_new_users - 0.5 * total_new_users) / math.sqrt(total_new_users * 0.25)
        pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    
    # Calculate percentages
    control_new_pct = (control_new_users / total_new_users * 100) if total_new_users > 0 else 0
    treatment_new_pct = (treatment_new_users / total_new_users * 100) if total_new_users > 0 else 0
    
    return jsonify({
        "control_group": CONTROL_GROUP,
        "treatment_group": TREATMENT_GROUP,
        "control_new_users": control_new_users,
        "treatment_new_users": treatment_new_users,
        "total_new_users": total_new_users,
        "control_new_percent": round(control_new_pct, 2),
        "treatment_new_percent": round(treatment_new_pct, 2),
        "difference": abs(control_new_users - treatment_new_users),
        "z_score": round(z, 3),
        "p_value_two_sided": round(pval, 4),
        "is_balanced": pval > 0.05,
        "note": "NEW USERS ONLY - First-time visitors, excluding returning users",
        "explanation": {
            "new_users": "Counts only genuinely new participants (first-time visitors)",
            "returning_users": "Users who reload/return are NOT counted here",
            "engagement": "Separate from engagement metrics (page views, sessions, etc.)",
            "purpose": "Shows true new user acquisition balance between groups"
        }
    })

@router.route('/track/engagement', methods=['POST'])
def track_engagement():
    """Track user engagement metrics"""
    try:
        data = request.get_json()
        pid = data.get('pid')
        group = data.get('group')
        page_views = data.get('page_views', 1)
        duration = data.get('duration', 0)
        interactions = data.get('interactions', 0)
        
        if not pid or not group:
            return jsonify({"status": "error", "message": "Missing pid or group"}), 400
        
        if group not in [CONTROL_GROUP, TREATMENT_GROUP]:
            return jsonify({"status": "error", "message": "Invalid group"}), 400
        
        _log_engagement(pid, group, page_views, duration, interactions)
        
        return jsonify({
            "status": "success",
            "message": "Engagement tracked successfully",
            "tracked": {
                "pid": pid,
                "group": group,
                "page_views": page_views,
                "duration": duration,
                "interactions": interactions
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to track engagement: {str(e)}"
        }), 500

@router.route('/track/pageview', methods=['POST'])
def track_pageview():
    """Track a simple page view"""
    try:
        data = request.get_json()
        pid = data.get('pid')
        group = data.get('group')
        
        if not pid or not group:
            return jsonify({"status": "error", "message": "Missing pid or group"}), 400
        
        if group not in [CONTROL_GROUP, TREATMENT_GROUP]:
            return jsonify({"status": "error", "message": "Invalid group"}), 400
        
        _log_engagement(pid, group, page_views=1)
        
        return jsonify({
            "status": "success",
            "message": "Page view tracked"
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to track page view: {str(e)}"
        }), 500

@router.route('/track/interaction', methods=['POST'])
def track_interaction():
    """Track user interactions (clicks, form submissions, etc.)"""
    try:
        data = request.get_json()
        pid = data.get('pid')
        group = data.get('group')
        interaction_type = data.get('type', 'click')
        interaction_value = data.get('value', None)
        
        if not pid or not group:
            return jsonify({"status": "error", "message": "Missing pid or group"}), 400
        
        if group not in [CONTROL_GROUP, TREATMENT_GROUP]:
            return jsonify({"status": "error", "message": "Invalid group"}), 400
        
        # Log the interaction
        _log_engagement(pid, group, interactions=1)
        
        # Also log detailed interaction data
        interaction_rec = {
            "ts": time.time(),
            "pid": pid,
            "group": group,
            "interaction_type": interaction_type,
            "interaction_value": interaction_value,
        }
        
        with _LOCK:
            with open(INTERACTIONS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(interaction_rec) + "\n")
        
        return jsonify({
            "status": "success",
            "message": "Interaction tracked successfully",
            "tracked": {
                "pid": pid,
                "group": group,
                "type": interaction_type,
                "value": interaction_value
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to track interaction: {str(e)}"
        }), 500

@router.route('/reset', methods=['POST'])
def reset_metrics():
    """Reset all metrics by clearing the assignment log file"""
    if not ENABLE_ADMIN_ROUTES:
        return jsonify({"error": "Admin routes are disabled in the public demo."}), 404
    try:
        with _LOCK:
            # Clear the assignment log file
            if os.path.exists(ASSIGN_LOG):
                # Create a backup before clearing
                backup_file = f"{ASSIGN_LOG}.backup.{int(time.time())}"
                os.rename(ASSIGN_LOG, backup_file)
                print(f"Backup created: {backup_file}")
            
            # Create empty log file
            with open(ASSIGN_LOG, "w", encoding="utf-8") as f:
                pass  # Create empty file
        
        return jsonify({
            "status": "success",
            "message": "Metrics reset successfully",
            "timestamp": time.time(),
            "backup_created": True if os.path.exists(backup_file) else False
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to reset metrics: {str(e)}",
            "timestamp": time.time()
        }), 500

@router.route('/reset', methods=['GET'])
def reset_info():
    """Show information about the reset endpoint"""
    if not ENABLE_ADMIN_ROUTES:
        return jsonify({"error": "Admin routes are disabled in the public demo."}), 404
    return jsonify({
        "message": "Use POST /reset to clear all metrics",
        "warning": "This will delete all assignment history!",
        "backup": "A backup file will be created automatically",
        "usage": "curl -X POST http://127.0.0.1:8050/reset",
        "current_groups": {
            "control_group": CONTROL_GROUP,
            "treatment_group": TREATMENT_GROUP
        },
        "other_options": {
            "reset_participant": f"POST /reset/participant/<pid> - Reset specific participant",
            "reset_group": f"POST /reset/group/<group> - Reset specific group ({CONTROL_GROUP} or {TREATMENT_GROUP})",
            "reset_with_seed": "POST /reset/seed/<seed> - Reset with specific random seed"
        }
    })

@router.route('/reset/participant/<pid>', methods=['POST'])
def reset_participant(pid):
    """Reset assignment for a specific participant ID"""
    if not ENABLE_ADMIN_ROUTES:
        return jsonify({"error": "Admin routes are disabled in the public demo."}), 404
    try:
        with _LOCK:
            if not os.path.exists(ASSIGN_LOG):
                return jsonify({"status": "error", "message": "No assignments to reset"}), 404
            
            # Read all lines and filter out the specified PID
            lines = []
            removed = False
            with open(ASSIGN_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get("pid") != pid:
                            lines.append(line)
                        else:
                            removed = True
                    except Exception:
                        lines.append(line)  # Keep malformed lines
            
            if removed:
                # Create backup
                backup_file = f"{ASSIGN_LOG}.backup.{int(time.time())}"
                os.rename(ASSIGN_LOG, backup_file)
                
                # Write filtered data back
                with open(ASSIGN_LOG, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                
                return jsonify({
                    "status": "success",
                    "message": f"Participant {pid} reset successfully",
                    "backup_created": backup_file
                }), 200
            else:
                return jsonify({
                    "status": "error", 
                    "message": f"Participant {pid} not found"
                }), 404
                
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to reset participant: {str(e)}"
        }), 500

@router.route('/reset/group/<group>', methods=['POST'])
def reset_group(group):
    """Reset all assignments for a specific group"""
    if not ENABLE_ADMIN_ROUTES:
        return jsonify({"error": "Admin routes are disabled in the public demo."}), 404
    if group not in [CONTROL_GROUP, TREATMENT_GROUP]:
        return jsonify({
            "status": "error", 
            "message": f"Group must be '{CONTROL_GROUP}' or '{TREATMENT_GROUP}'"
        }), 400
    
    try:
        with _LOCK:
            if not os.path.exists(ASSIGN_LOG):
                return jsonify({"status": "error", "message": "No assignments to reset"}), 404
            
            # Read all lines and filter out the specified group
            lines = []
            removed_count = 0
            with open(ASSIGN_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get("group") != group:
                            lines.append(line)
                        else:
                            removed_count += 1
                    except Exception:
                        lines.append(line)  # Keep malformed lines
            
            if removed_count > 0:
                # Create backup
                backup_file = f"{ASSIGN_LOG}.backup.{int(time.time())}"
                os.rename(ASSIGN_LOG, backup_file)
                
                # Write filtered data back
                with open(ASSIGN_LOG, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                
                return jsonify({
                    "status": "success",
                    "message": f"Group {group} reset successfully",
                    "participants_removed": removed_count,
                    "backup_created": backup_file
                }), 200
            else:
                return jsonify({
                    "status": "error", 
                    "message": f"No participants found in group {group}"
                }), 404
                
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to reset group: {str(e)}"
        }), 500

@router.route('/reset/seed/<int:seed>', methods=['POST'])
def reset_with_seed(seed):
    """Reset all assignments and set a new random seed for reproducible testing"""
    if not ENABLE_ADMIN_ROUTES:
        return jsonify({"error": "Admin routes are disabled in the public demo."}), 404
    try:
        with _LOCK:
            # Set the random seed
            random.seed(seed)
            
            # Create backup of current assignments
            if os.path.exists(ASSIGN_LOG):
                backup_file = f"{ASSIGN_LOG}.backup.{int(time.time())}"
                os.rename(ASSIGN_LOG, backup_file)
            else:
                backup_file = None
            
            # Create empty log file
            with open(ASSIGN_LOG, "w", encoding="utf-8") as f:
                pass
            
            return jsonify({
                "status": "success",
                "message": f"Reset with seed {seed} successful",
                "seed": seed,
                "backup_created": backup_file if backup_file else False,
                "note": "Random seed set for reproducible assignments"
            }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to reset with seed: {str(e)}"
        }), 500

@router.route('/health')
def health():
    return 'OK', 200

@router.route('/debug')
def debug_session():
    """Debug endpoint to show current session and assignment info"""
    if not ENABLE_ADMIN_ROUTES:
        return jsonify({"error": "Debug routes are disabled in the public demo."}), 404
    pid = session.get("pid")
    group = session.get("group")
    existing_group = _get_existing_assignment(pid) if pid else None

    return jsonify({
        "has_session_pid": bool(pid),
        "session_group": group,
        "logged_group": existing_group,
        "group_mapping": {
            "control_group": CONTROL_GROUP,
            "treatment_group": TREATMENT_GROUP
        },
        "analysis": {
            "has_pid": bool(pid),
            "has_session_group": bool(group),
            "has_logged_group": bool(existing_group),
            "groups_match": group == existing_group if group and existing_group else None,
            "should_redirect_to": f"/{existing_group}/" if existing_group else "No logged group",
            "pid_source": "session" if pid else "none"
        }
    })

@router.route('/blocked-randomization')
def blocked_randomization_info():
    """Show information about blocked randomization configuration and status"""
    control_count, treatment_count = _load_counts()
    total = control_count + treatment_count
    
    # Calculate current block information
    current_block = (total // BLOCK_SIZE) + 1
    position_in_block = total % BLOCK_SIZE
    participants_in_current_block = position_in_block if position_in_block > 0 else BLOCK_SIZE
    
    # Calculate expected balance at end of current block
    if position_in_block == 0:
        # At start of block, should be perfectly balanced
        expected_control = total
        expected_treatment = total
        balance_status = "Perfectly balanced"
    else:
        # Within block, calculate expected final balance
        remaining_in_block = BLOCK_SIZE - position_in_block
        expected_control = control_count + (remaining_in_block // 2)
        expected_treatment = treatment_count + (remaining_in_block // 2)
        if remaining_in_block % 2 == 1:
            # Odd number remaining, one group gets extra
            if control_count < treatment_count:
                expected_control += 1
            else:
                expected_treatment += 1
        balance_status = f"Will be balanced after {remaining_in_block} more participants"
    
    return jsonify({
        "blocked_randomization_enabled": ENABLE_BLOCKED_RANDOMIZATION,
        "block_size": BLOCK_SIZE,
        "current_status": {
            "total_participants": total,
            "current_block": current_block,
            "position_in_current_block": position_in_block,
            "participants_in_current_block": participants_in_current_block,
            "remaining_in_current_block": BLOCK_SIZE - position_in_block if position_in_block > 0 else 0
        },
        "current_counts": {
            "control": control_count,
            "treatment": treatment_count,
            "difference": abs(control_count - treatment_count)
        },
        "expected_balance": {
            "at_end_of_block": {
                "control": expected_control,
                "treatment": expected_treatment,
                "difference": abs(expected_control - expected_treatment)
            },
            "status": balance_status
        },
        "block_patterns": {
            "block_size_2": ["AB", "BA"],
            "block_size_4": ["AABB", "ABAB", "ABBA", "BAAB", "BABA", "BBAA"],
            "block_size_6": ["AAABBB", "AABABB", "AABBAB", "AABBBA", "ABAABB", "ABABAB", 
                            "ABABBA", "ABBAAB", "ABBABA", "ABBBAA", "BAAABB", "BAABAB",
                            "BAABBA", "BABAAB", "BABABA", "BABBAA", "BBAAAB", "BBAABA",
                            "BBABAA", "BBBAAA"]
        },
        "explanation": {
            "what_it_does": "Ensures perfect balance every N participants (where N = block size)",
            "maximum_imbalance": f"Maximum imbalance is {BLOCK_SIZE // 2} participants",
            "unpredictability": "Within each block, order is randomized for unpredictability",
            "best_for": "Small sample sizes (N < 50) where balance is critical"
        }
    })

@router.route('/debug/pid/<pid>')
def debug_pid(pid):
    """Debug endpoint to check a specific PID's assignment history"""
    if not ENABLE_ADMIN_ROUTES:
        return jsonify({"error": "Debug routes are disabled in the public demo."}), 404
    try:
        # Check assignments log
        assignments = []
        if os.path.exists(ASSIGN_LOG):
            with open(ASSIGN_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get("pid") == pid:
                            assignments.append(rec)
                    except Exception:
                        continue
        
        # Check new users log
        new_user_records = []
        if os.path.exists(NEW_USERS_LOG):
            with open(NEW_USERS_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get("pid") == pid:
                            new_user_records.append(rec)
                    except Exception:
                        continue
        
        # Check engagement log
        engagement_records = []
        if os.path.exists(ENGAGEMENT_LOG):
            with open(ENGAGEMENT_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get("pid") == pid:
                            engagement_records.append(rec)
                    except Exception:
                        continue
        
        return jsonify({
            "pid": pid,
            "assignments": assignments,
            "new_user_records": new_user_records,
            "engagement_records": engagement_records,
            "summary": {
                "total_assignments": len(assignments),
                "is_new_user": len(new_user_records) > 0,
                "total_engagement": len(engagement_records),
                "current_group": assignments[-1].get("group") if assignments else None
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "pid": pid
        }), 500

# Mount the two apps
application = DispatcherMiddleware(router, {
    f'/{CONTROL_GROUP}': get_control_app(),
    f'/{TREATMENT_GROUP}': get_treatment_app(),
})

if __name__ == '__main__':
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8050"))
    print(f"Router app running on http://{host}:{port}/")
    print(f"Check balance at: http://{host}:{port}/metrics")
    run_simple(host, port, application, use_reloader=False, use_debugger=False)
