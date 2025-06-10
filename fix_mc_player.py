import re

def fix_mc_player():
    with open('agents/MC_player.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 移除重複的常量定義
    pattern = r'# 先定義基本常量\n\s+CARD_RANKS = .*?\n\s+STATIC_DECK = .*?\n\s+# 先定義基本常量'
    replacement = '# 先定義基本常量'
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open('agents/MC_player.py', 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == '__main__':
    fix_mc_player() 