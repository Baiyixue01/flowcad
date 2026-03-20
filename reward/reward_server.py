from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
import traceback

app = FastAPI()


# ===== 请求和响应格式 =====
class RewardRequest(BaseModel):
    completions: list[str]
    prompts: list[str] | None = None
    metas: list[dict[str, Any]] | None = None


class RewardResponse(BaseModel):
    rewards: list[float]
    ok: bool = True
    error: str | None = None


# ===== 你的重型资源：启动时加载一次 =====
# 比如 CAD evaluator / 仿真器 / 模型 / 几何分析器
# 不要放到每次请求里初始化
class HeavyRewardEngine:
    def __init__(self):
        # 在这里加载重依赖
        # 例如：
        # self.evaluator = CADMetricEvaluator(...)
        # self.sim_env = SomeEnv(...)
        print("HeavyRewardEngine initialized.")

    def score_one(self, completion: str, prompt: str | None = None, meta: dict | None = None) -> float:
        """
        这里替换成你的真实 reward 逻辑。
        返回 float。
        """
        try:
            text = completion.strip()

            # ===== 示例逻辑，先占位 =====
            # 真实情况你可以：
            # 1. 执行代码
            # 2. 构建 CAD
            # 3. 检查是否报错
            # 4. 计算几何指标
            # 5. 返回 reward
            if not text:
                return -1.0

            reward = 0.0

            # 例子：长度太短惩罚
            if len(text) < 20:
                reward -= 0.5

            # 例子：出现 result_0 奖励一点
            if "result_" in text:
                reward += 0.2

            # 例子：假设可执行再给奖励
            reward += 0.5

            return float(reward)

        except Exception:
            return -2.0


engine = HeavyRewardEngine()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reward", response_model=RewardResponse)
def compute_reward(req: RewardRequest):
    try:
        rewards = []

        prompts = req.prompts or [None] * len(req.completions)
        metas = req.metas or [None] * len(req.completions)

        if not (len(req.completions) == len(prompts) == len(metas)):
            return RewardResponse(
                rewards=[],
                ok=False,
                error="Length mismatch among completions/prompts/metas",
            )

        for completion, prompt, meta in zip(req.completions, prompts, metas):
            score = engine.score_one(completion=completion, prompt=prompt, meta=meta)
            rewards.append(float(score))

        return RewardResponse(rewards=rewards, ok=True)

    except Exception as e:
        traceback.print_exc()
        return RewardResponse(rewards=[], ok=False, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("reward_server:app", host="0.0.0.0", port=8005)