from typing import Any
import importlib
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
        self.remote_reward_fn = None
        for module_name in ("reward.reward_fun", "reward_fun"):
            try:
                module = importlib.import_module(module_name)
                self.remote_reward_fn = getattr(module, "reward_fn", None)
                if self.remote_reward_fn is not None:
                    print(f"HeavyRewardEngine initialized with {module_name}.reward_fn")
                    return
            except Exception:
                traceback.print_exc()

        print("HeavyRewardEngine initialized with fallback demo scorer.")

    def score_batch(
        self,
        completions: list[str],
        prompts: list[str] | None = None,
        metas: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        if self.remote_reward_fn is not None:
            kwargs: dict[str, list[Any]] = {}
            if metas:
                for meta in metas:
                    for key, value in (meta or {}).items():
                        kwargs.setdefault(key, []).append(value)

                # 保证每个 key 与 batch 对齐
                batch_size = len(completions)
                for key, val_list in kwargs.items():
                    if len(val_list) < batch_size:
                        kwargs[key] = val_list + [None] * (batch_size - len(val_list))

            scores = self.remote_reward_fn(
                prompts=prompts or ["" for _ in completions],
                completions=completions,
                **kwargs,
            )
            return [float(x) for x in scores]

        scores = []
        for completion, prompt, meta in zip(completions, prompts or [], metas or []):
            scores.append(self.score_one(completion=completion, prompt=prompt, meta=meta))
        if not scores:
            scores = [self.score_one(completion=c) for c in completions]
        return [float(x) for x in scores]

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
        prompts = req.prompts or [None] * len(req.completions)
        metas = req.metas or [None] * len(req.completions)

        if not (len(req.completions) == len(prompts) == len(metas)):
            return RewardResponse(
                rewards=[],
                ok=False,
                error="Length mismatch among completions/prompts/metas",
            )

        rewards = engine.score_batch(
            completions=req.completions,
            prompts=prompts,
            metas=metas,
        )

        return RewardResponse(rewards=rewards, ok=True)

    except Exception as e:
        traceback.print_exc()
        return RewardResponse(rewards=[], ok=False, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("reward_server:app", host="0.0.0.0", port=8005)
