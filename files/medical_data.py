# Define medical data frames for CKD
medical_data_ckd = {
    "medical_df1_ckd": """
    ## 患者信息：
    ### 肾衰竭风险方程参数：
    1. 年龄: 78岁
    2. 性别: 男性
    3. eGFR: 15 mL/min/1.73 m²
    4. 血清肌酐: 5.2 mg/dL
    5. 尿白蛋白/肌酐比率（ACR）: 1200 mg/g
    6. 血清钙: 8.3 mg/dL
    7. 血清磷: 5.5 mg/dL
    8. 血红蛋白（Hb）: 9.5 g/dL
    9. 收缩压: 160 mmHg
    """,
    "medical_df2_ckd": """
    ## 患者信息：
    ### 肾衰竭风险方程参数：
    1. 年龄：58岁
    2. 性别：男性
    3. eGFR：24 mL/min/1.73 m²
    4. 血清肌酐：2.6 mg/dL
    5. 尿白蛋白/肌酐比率（ACR）：530 mg/g
    6. 血清钙：8.9 mg/dL
    7. 血清磷：4.9 mg/dL
    8. 血红蛋白（Hb）：10.5 g/dL
    9. 收缩压：148 mmHg
    """,
    "medical_df3_ckd": """
    ## 患者信息：
    ### 肾衰竭风险方程参数：
    1. 年龄: 62岁
    2. 性别: 男性
    3. eGFR: 8 mL/min/1.73 m²
    4. 血清肌酐: 6.5 mg/dL 
    5. 尿白蛋白/肌酐比率（ACR）: 1500 mg/g
    6. 血清钙: 7.8 mg/dL
    7. 血清磷: 6.2 mg/dL
    8. 血红蛋白（Hb）: 8.2 g/dL
    9. 收缩压: 180 mmHg
    10. 症状: 恶心、呕吐、乏力、呼吸困难，伴有水肿
    """
}

# Define medical data frames for dialysis
medical_data_dialysis = {
    "medical_df1_dialysis": """
    ### 透析前信息：
    - 体重： 72.5 公斤
    - 血压： 145/85 mmHg
    - 脉搏： 78 次/分钟
    - 体温： 36.6°C
    - 干体重： 70 公斤

    ### 透析详情：
    - 透析开始时间： 09:00
    - 透析时长： 4 小时
    - 血流速率： 300 mL/分钟
    - 透析液流速： 500 mL/分钟
    - 使用透析器： FX60型
    - 肝素剂量： 3000 单位
    - 目标超滤量： 2.5 升

    ### 透析中监测：
    - 血压读数： 130/80, 120/75, 115/70
    - 脉搏读数： 72, 70, 68 次/分钟
    - 实际超滤量： 2.3 升
    - 并发症： 无

    ### 透析后信息：
    - 体重： 70.2 公斤
    - 血压： 110/65 mmHg
    - 脉搏： 66 次/分钟
    - 总超滤量： 2.3 升
    - 症状： 无
    """,

    "medical_df2_dialysis": """
    ### 透析前信息：
    - 体重： 75 公斤
    - 血压： 150/90 mmHg
    - 脉搏： 80 次/分钟
    - 体温： 36.7°C
    - 干体重： 72 公斤

    ### 透析详情：
    - 透析开始时间： 08:30
    - 透析时长： 4 小时
    - 血流速率： 320 mL/分钟
    - 透析液流速： 500 mL/分钟
    - 使用透析器： FX80型
    - 肝素剂量： 3500 单位
    - 目标超滤量： 3 升

    ### 透析中监测：
    - 血压读数： 135/85, 125/80, 120/75
    - 脉搏读数： 75, 72, 70 次/分钟
    - 实际超滤量： 2.8 升
    - 并发症： 无

    ### 透析后信息：
    - 体重： 72.2 公斤
    - 血压： 115/70 mmHg
    - 脉搏： 68 次/分钟
    - 总超滤量： 2.8 升
    - 症状： 无
    """,

    "medical_df3_dialysis": """
    ### 透析前信息：
    - 体重： 68 公斤
    - 血压： 140/85 mmHg
    - 脉搏： 76 次/分钟
    - 体温： 36.5°C
    - 干体重： 66 公斤

    ### 透析详情：
    - 透析开始时间： 10:00
    - 透析时长： 4 小时
    - 血流速率： 280 mL/分钟
    - 透析液流速： 500 mL/分钟
    - 使用透析器： FX50型
    - 肝素剂量： 3000 单位
    - 目标超滤量： 2 升

    ### 透析中监测：
    - 血压读数： 130/80, 120/75, 115/70
    - 脉搏读数： 72, 70, 68 次/分钟
    - 实际超滤量： 1.8 升
    - 并发症： 无

    ### 透析后信息：
    - 体重： 66.2 公斤
    - 血压： 110/65 mmHg
    - 脉搏： 66 次/分钟
    - 总超滤量： 1.8 升
    - 症状： 无
    """
}

CKD_prompt = """
    数据如下：
    {medical_df}

    请根据以上数据撰写患者的医疗评估报告。请使用简单的文本格式，不要使用粗体或标题格式，只需使用编号和项目符号。

    排版重点：
    • 不使用粗体或标题格式
    • 使用简单的编号和项目符号格式
    • 简洁清晰地呈现每个评估和建议

    格式如下：

    患者医疗评估

    1. ESRD风险: 根据肾衰竭风险方程 (KFRE) 计算，患者在两年内进展至终末期肾病 (ESRD) 的风险约为 **{ESRD_risk_percentage}%**。  
    eGFR 为 {eGFR_value} mL/min/1.73 m²，ACR 为 {ACR_value} mg/g，提示进展风险较高。

    2. 临床问题:  
    - 高血压: 收缩压 {systolic_bp} mmHg，需控制。  
    - 贫血: 血红蛋白 {hb_value} g/dL。  
    - 钙磷代谢紊乱: 钙和磷水平不正常，需关注骨骼健康。

    3. 治疗建议:  可以梳理但是不要过多，用项目符号控制在3-5条
    控制高血压、贫血及钙磷代谢，改善饮食和生活方式，定期复查 eGFR 和 ACR 以监测病情进展。
"""

dialysis_prompt_old = """
    数据如下：
    {medical_df}

    请根据以下格式撰写患者的透析评估报告，并根据患者的具体值提供动态的分析和管理建议.请使用简单的文本格式，不要使用粗体或标题格式，只需使用编号和项目符号。
    
    排版重点：
	• 不使用粗体或标题格式
	• 使用简单的编号和项目符号格式
	• 简洁清晰地呈现每个评估和建议

    患者透析评估报告

    1. 透析前信息
    - 体重: {medical_df.splitlines()[2].split('： ')[1]}
    - 血压: {medical_df.splitlines()[3].split('： ')[1]}
    - 脉搏: {medical_df.splitlines()[4].split('： ')[1]}
    - 体温: {medical_df.splitlines()[5].split('： ')[1]}
    - 干体重: {medical_df.splitlines()[6].split('： ')[1]}

    2. 透析详情
    - 透析开始时间: {medical_df.splitlines()[9].split('： ')[1]}
    - 透析时长: {medical_df.splitlines()[10].split('： ')[1]}
    - 血流速率: {medical_df.splitlines()[11].split('： ')[1]}
    - 透析液流速: {medical_df.splitlines()[12].split('： ')[1]}
    - 使用透析器: {medical_df.splitlines()[13].split('： ')[1]}
    - 肝素剂量: {medical_df.splitlines()[14].split('： ')[1]}
    - 目标超滤量: {medical_df.splitlines()[15].split('： ')[1]}

    3. 透析中监测
    - 血压读数: {medical_df.splitlines()[18].split('： ')[1]}
    - 脉搏读数: {medical_df.splitlines()[19].split('： ')[1]}
    - 实际超滤量: {medical_df.splitlines()[20].split('： ')[1]}
    - 并发症: {medical_df.splitlines()[21].split('： ')[1]}

    4. 透析后信息
    - 体重: {medical_df.splitlines()[24].split('： ')[1]}
    - 血压: {medical_df.splitlines()[25].split('： ')[1]}
    - 脉搏: {medical_df.splitlines()[26].split('： ')[1]}
    - 总超滤量: {medical_df.splitlines()[27].split('： ')[1]}
    - 症状: {medical_df.splitlines()[28].split('： ')[1]}

    5. 治疗建议
    - 血压管理: 
    - 贫血管理: 
    - 透析方案调整: 
    - 营养管理: 

    6. 随访建议
    - 定期随访：
    - 营养咨询：

    7. 患者教育
    - 饮食: 
    - 运动: 
    - 透析管理与计划: 

    8. 简要分析：
    - 透析前信息: 体重 {medical_df.splitlines()[2].split('： ')[1]}, 血压 {medical_df.splitlines()[3].split('： ')[1]}, 脉搏 {medical_df.splitlines()[4].split('： ')[1]}, 体温 {medical_df.splitlines()[5].split('： ')[1]}, 干体重 {medical_df.splitlines()[6].split('： ')[1]}
    - 透析详情: 透析开始时间 {medical_df.splitlines()[9].split('： ')[1]}, 透析时长 {medical_df.splitlines()[10].split('： ')[1]}, 血流速率 {medical_df.splitlines()[11].split('： ')[1]}, 透析液流速 {medical_df.splitlines()[12].split('： ')[1]}, 使用透析器 {medical_df.splitlines()[13].split('： ')[1]}, 肝素剂量 {medical_df.splitlines()[14].split('： ')[1]}, 目标超滤量 {medical_df.splitlines()[15].split('： ')[1]}
    - 透析中监测: 血压读数 {medical_df.splitlines()[18].split('： ')[1]}, 脉搏读数 {medical_df.splitlines()[19].split('： ')[1]}, 实际超滤量 {medical_df.splitlines()[20].split('： ')[1]}, 并发症 {medical_df.splitlines()[21].split('： ')[1]}
    - 透析后信息: 体重 {medical_df.splitlines()[24].split('： ')[1]}, 血压 {medical_df.splitlines()[25].split('： ')[1]}, 脉搏 {medical_df.splitlines()[26].split('： ')[1]}, 总超滤量 {medical_df.splitlines()[27].split('： ')[1]}, 症状 {medical_df.splitlines()[28].split('： ')[1]}
"""

dialysis_prompt = """
    数据如下：
    {medical_df}

    请根据以下格式撰写患者的透析评估报告，并根据患者的具体值提供动态的分析和管理建议.请使用简单的文本格式，不要使用粗体或标题格式，只需使用编号和项目符号。
    
    排版重点：
	• 不使用粗体或标题格式
	• 使用简单的编号和项目符号格式
	• 简洁清晰地呈现每个评估和建议

    患者透析评估报告
    1. 简要分析：
    - 透析前信息: 体重, 血压, 脉搏, 体温, 干体重
    - 透析详情: 透析开始时间, 透析时长, 血流速率, 透析液流速, 使用透析器, 肝素剂量, 目标超滤量
    - 透析中监测: 血压读数, 脉搏读数, 实际超滤量, 并发症
    - 透析后信息: 体重, 血压, 脉搏, 总超滤量, 症状

    2. 治疗建议
    - 血压管理: 
    - 贫血管理: 
    - 透析方案调整: 
    - 营养管理: 

    3. 其他建议:  可以梳理但是不要过多，用项目符号控制在3条
    随访建议 营养咨询 患者教育 
"""


