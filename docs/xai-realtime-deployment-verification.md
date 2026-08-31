# xAI Realtime Voice 部署验收单

本文用于验证 `open-xiaoai-bridge` 的 xAI Grok Realtime Voice 部署是否正常，覆盖：

- Realtime WebSocket 与 Token 鉴权
- xAI Session 初始化
- Rust/PyO3 与 Sonora AEC3
- 音箱唤醒、连续对话和语音插话
- 退出、超时和“小爱同学”中断
- AEC 失败时的半双工降级

## 1. 验收前准备

进入项目目录：

```bash
cd /path/to/open-xiaoai-bridge
```

通过环境变量提供 Token，避免将密钥写入命令历史或提交到 Git：

```bash
read -rsp 'XAI_API_KEY: ' XAI_API_KEY
export XAI_API_KEY
echo

export XAI_LIVE_API_URL='wss://your-domain.example/v1/realtime?model=grok-voice-latest'
```

如果使用 systemd、Supervisor 等进程管理器，应将 `XAI_API_KEY` 和 `XAI_ENABLE=1` 配置到对应服务的环境变量中。

`config.py` 建议只保留非敏感配置：

```python
"xai": {
    "api_url": "wss://your-domain.example/v1/realtime?model=grok-voice-latest",
    "api_key": "",  # 推荐通过 XAI_API_KEY 提供
    "voice": "ara",
    "instructions": "你是一个有帮助的语音助手，请用简洁口语中文回答。",
    "sample_rate": 16000,
    "exit_keywords": ["退出", "停止", "再见"],
    "idle_timeout": 20,
    "aec": True,
    "aec_delay_ms": 150,
    "greeting": True,
}
```

公网部署必须使用 `wss://`，避免 Token、音频和对话内容通过明文网络传输。

## 2. 检查 curl WebSocket 支持

```bash
curl --version
```

通过标准：`Protocols:` 中包含 `ws` 和 `wss`。

macOS 自带的 curl 可能没有启用 WebSocket。可以使用 Homebrew curl：

```bash
brew install curl
echo 'export PATH="/opt/homebrew/opt/curl/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
hash -r
```

再次执行 `curl --version`，确认实际调用的是 Homebrew curl，并且协议列表包含 `ws wss`。

## 3. 验证 Realtime WebSocket

```bash
curl -sS -N --max-time 5 \
  -H "Authorization: Bearer $XAI_API_KEY" \
  "$XAI_LIVE_API_URL" | jq -c '.type'
```

通过标准：至少依次收到以下事件：

```text
"session.created"
"conversation.created"
"ping"
```

WebSocket 是长连接，5 秒后出现以下信息属于预期行为，不代表失败：

```text
curl: (28) Operation timed out
```

常见失败：

| 现象 | 可能原因 |
|---|---|
| `401` / `403` | Token 无效、过期或代理未正确转发鉴权头 |
| `Connection refused` | 服务未启动、端口未监听或防火墙拦截 |
| TLS/证书错误 | 域名证书无效、证书链不完整或反向代理配置错误 |
| 只收到 HTTP 页面 | 反向代理没有正确处理 WebSocket Upgrade |

## 4. 运行项目实时连通性测试

仓库提供了不会触发开场白和音频播放的实时测试：

```bash
XAI_LIVE_TEST=1 \
XAI_LIVE_API_URL="$XAI_LIVE_API_URL" \
uv run python -m unittest tests/test_xai_live_connectivity.py -v
```

通过标准：

```text
test_connect_and_configure_session ... ok
```

该测试会验证：

- WebSocket 与 Bearer Token 鉴权
- `conversation.created`
- 客户端发送 `session.update`
- 服务端返回 `session.updated`
- 获得有效的 `conversation_id`

## 5. 验证 Rust/PyO3 与 Sonora AEC3

检查 Python 是否能加载 Rust 原生扩展：

```bash
uv run python -c \
  'import open_xiaoai_server; print(open_xiaoai_server.AecProcessor)'
```

通过标准：输出 `AecProcessor` 类型，并且没有 `ImportError`。

如果导入失败，在目标机器上重新编译安装：

```bash
uv run maturin develop --release --manifest-path native/Cargo.toml
```

N100 通常是 `x86_64`，必须使用目标机器本地编译的扩展或对应架构的 wheel，不能使用 macOS ARM64 wheel。

可以额外运行 native 音频测试：

```bash
uv run python -m unittest tests/test_xai_native_audio.py -v
```

通过标准：AEC 回声抑制、帧尺寸校验和播放 token 测试全部通过。

## 6. 启动服务

```bash
XAI_ENABLE=1 uv run main.py
```

如果服务由 systemd 管理，可使用对应服务日志观察运行状态：

```bash
journalctl -fu your-service-name
```

启动阶段不应出现：

- `xAI API key 未配置`
- `No module named open_xiaoai_server`
- KWS/VAD 模型加载失败
- 音频服务端口被占用

## 7. 音箱功能验收

| 编号 | 项目 | 操作 | 通过标准 |
|---|---|---|---|
| 1 | KWS 唤醒 | 说“你好 grok” | 播放“Grok 来了”，随后进入实时对话 |
| 2 | 小爱路由 | 对小爱说“召唤 grok” | 中断小爱当前操作并进入 Grok 对话 |
| 3 | 开场白 | 等待 Session 初始化 | Grok 主动进行简短问候 |
| 4 | 中文对话 | 问“用一句话介绍你自己” | 收到清晰的中文语音回答 |
| 5 | 连续对话 | 连续询问三个问题 | 无需重复唤醒，问题均能得到回答 |
| 6 | AEC 稳定性 | AI 回答时保持安静 | AI 不会被自己的扬声器声音反复打断 |
| 7 | 语音插话 | AI 说话时提出新问题 | 当前回答迅速停止，并处理新的问题 |
| 8 | 退出关键词 | 说“退出”“停止”或“再见” | 会话退出并执行 `after_wakeup` |
| 9 | 空闲退出 | 进入会话后保持安静 | 达到 `idle_timeout` 后自动退出 |
| 10 | 小爱中断 | AI 说话时喊“小爱同学” | Grok 立即停止，小爱恢复响应 |
| 11 | 二次唤醒 | 退出后再次说“你好 grok” | 能建立新的 Realtime 会话 |

## 8. 日志验收

正常会话应能看到类似日志：

```text
进入 Grok Voice 实时对话
Sonora AEC3 已启用，delay=150ms
退出 Grok Voice 实时对话
```

退出关键词应出现：

```text
检测到退出关键词
```

空闲超时应出现：

```text
等待用户说话超时
```

以下日志需要排查：

```text
AEC 不可用，当前会话降级为半双工
播放 PCM 失败
上行音频失败
WebSocket closed
```

其中 AEC 降级不会直接导致会话失败，但意味着当前会话不再支持真正的播放期间插话。

## 9. 半双工降级验收

临时修改配置并重启：

```python
"aec": False,
```

验证以下行为：

- 仍然能够正常听取用户语音并返回语音回答
- AI 播放期间暂停上传麦克风音频
- AI 不会被自身回声打断
- 回答结束后能够恢复监听
- AI 播放期间无法插话属于预期行为

测试完成后恢复：

```python
"aec": True,
```

## 10. 回归检查

如果部署同时启用了其他后端，需要确认原有路径未受影响：

- “你好小智”仍能进入小智会话
- “龙虾”或“召唤龙虾”仍能进入 OpenClaw
- 小爱自身的播放、连续对话和中断仍正常
- xAI 退出后 KWS 能继续收到音频

运行完整测试：

```bash
uv run python -m unittest discover -s tests
```

OpenClaw 和 xAI 的 live 测试默认跳过，只有显式设置对应环境变量时才会访问真实服务。

## 11. 最终验收记录

- [ ] WSS、TLS 和 Token 鉴权正常
- [ ] 收到 `session.created`、`conversation.created` 和 `ping`
- [ ] `session.update` / `session.updated` 正常
- [ ] Rust/PyO3 扩展正常加载
- [ ] Sonora AEC3 测试通过
- [ ] KWS 和小爱指令均能进入 Grok Voice
- [ ] 中文语音对话正常
- [ ] 连续对话正常
- [ ] AI 不会被自身扬声器声音打断
- [ ] 播放期间可以语音插话
- [ ] 退出词、空闲超时和“小爱同学”中断正常
- [ ] 退出后可以再次唤醒
- [ ] `aec=False` 半双工降级正常
- [ ] 小智、OpenClaw 和小爱原有路径未受影响
- [ ] Token 未写入 Git 或公开日志

验收信息：

```text
验收日期：
机器/架构：
系统版本：
Git commit：
xAI endpoint：
模型：
音箱型号：
AEC delay：
验收人：
备注：
```
