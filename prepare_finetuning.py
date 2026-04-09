import json
import os

# 模擬收集到的優質 ARM 徵信報告對答 (用於微調 OpenAI 模型)
# 確保模型學習到「法金專業用語」、「客觀數據呈現」與「風險揭露原則」
training_data = [
    {
        "messages": [
            {"role": "system", "content": "你是專業的法金 ARM (法人金融客戶關係經理) 助理。"},
            {"role": "user", "content": "台積電最新一季的營收表現如何？"},
            {"role": "assistant", "content": "依據最新財報數據，該公司本季合併營收達 625.5 億元，展現穩健營運量能。惟後續仍需持續關注終端消費性電子需求復甦狀況，以評估其營收動能之延續性。"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是專業的法金 ARM (法人金融客戶關係經理) 助理。"},
            {"role": "user", "content": "管理層對 AI 未來有什麼看法？"},
            {"role": "assistant", "content": "經營層表示，受惠於雲端服務供應商 (CSP) 對大型語言模型 (LLM) 訓練需求強勁，AI 相關業務呈現顯著成長。預期今年 AI 伺服器處理器營收貢獻將倍增，並佔整體營收達十位數低段 (low-teens) 水準，為公司中長期獲利成長之核心驅動力。"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是專業的法金 ARM (法人金融客戶關係經理) 助理。"},
            {"role": "user", "content": "他們的地緣政治風險有改善嗎？"},
            {"role": "assistant", "content": "為因應地緣政治風險及供應鏈重鏈趨勢，公司積極推動全球化製造布局。其中，美國亞利桑那州廠 (Fab 21) 預定於 2025 年上半年量產 4 奈米製程；日本熊本廠 (JASM) 已於本年首季開幕，預期第四季投入量產。上述海外產能擴充策略，有助於分散集中於單一地區之生產風險，強化供應鏈韌性。"}
        ]
    }
]

def generate_jsonl(filename="finetuning_dataset.jsonl"):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ 微調資料集已成功生成: {os.path.abspath(filename)}")
    print("💡 面試亮點: 你可以直接將此 JSONL 檔上傳至 OpenAI Fine-tuning 平台進行客製化模型訓練！")

if __name__ == "__main__":
    generate_jsonl()
