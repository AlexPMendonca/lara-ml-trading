from datetime import datetime, timedelta
import math


class CooldownAuditState:
    def __init__(self):
        self.last_exit_time = None
        self.last_result = "NONE"
        self.last_result_pts = 0.0
        self.loss_streak = 0
        self.cooldown_release_time = None


def _safe_parse_dt(x):
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    s = str(x).strip()
    if not s:
        return None

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass

    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _fmt_dt(dt_obj):
    if dt_obj is None:
        return "NONE"
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")


def _join_blockers(*parts):
    out = []
    seen = set()
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if not s or s == "NONE":
            continue
        for token in s.split("|"):
            token = token.strip()
            if token and token not in seen:
                seen.add(token)
                out.append(token)
    return "|".join(out) if out else "NONE"


def ensure_cd_audit_state(ctx):
    if not hasattr(ctx, "_cooldown_audit_state") or getattr(ctx, "_cooldown_audit_state") is None:
        setattr(ctx, "_cooldown_audit_state", CooldownAuditState())
    return getattr(ctx, "_cooldown_audit_state")


def register_closed_trade_for_audit(
    ctx,
    exit_time,
    result_pts,
    result_label=None,
    cooldown_barras=3,
    candle_seconds=120,
    cooldown_apenas_apos_loss=True,
):
    st = ensure_cd_audit_state(ctx)

    exit_dt = _safe_parse_dt(exit_time)
    pts = float(result_pts or 0)

    if result_label is None:
        if pts > 0:
            result_label = "WIN"
        elif pts < 0:
            result_label = "LOSS"
        else:
            result_label = "EOD"

    st.last_exit_time = exit_dt
    st.last_result = str(result_label).upper()
    st.last_result_pts = pts

    if st.last_result == "LOSS":
        st.loss_streak += 1
    elif st.last_result == "WIN":
        st.loss_streak = 0

    should_start_cooldown = (st.last_result == "LOSS") if cooldown_apenas_apos_loss else (st.last_result in ("LOSS", "WIN", "EOD"))

    if should_start_cooldown and exit_dt is not None:
        st.cooldown_release_time = exit_dt + timedelta(seconds=int(cooldown_barras) * int(candle_seconds))
    else:
        st.cooldown_release_time = None


def build_cooldown_audit_fields(
    ctx,
    now_time,
    signal_elegivel=False,
    bloqueado_por_existente="NONE",
    candle_seconds=120,
):
    st = ensure_cd_audit_state(ctx)
    now_dt = _safe_parse_dt(now_time)

    cooldown_ativo = 0
    cooldown_restante_barras = 0
    cooldown_restante_min = 0

    if st.cooldown_release_time is not None and now_dt is not None and now_dt < st.cooldown_release_time:
        cooldown_ativo = 1
        remaining_sec = max(0, int((st.cooldown_release_time - now_dt).total_seconds()))
        cooldown_restante_barras = int(math.ceil(remaining_sec / float(candle_seconds)))
        cooldown_restante_min = int(math.ceil(remaining_sec / 60.0))

    entrada_bloqueada_cooldown = 1 if (signal_elegivel and cooldown_ativo) else 0

    bloqueado_por = bloqueado_por_existente if bloqueado_por_existente else "NONE"
    if cooldown_ativo:
        bloqueado_por = _join_blockers(bloqueado_por, "cooldown")
    else:
        bloqueado_por = _join_blockers(bloqueado_por)

    if entrada_bloqueada_cooldown:
        motivo = f"cooldown apos {st.last_result} pts={st.last_result_pts:.0f} libera={_fmt_dt(st.cooldown_release_time)}"
    elif "sem_sinal" in bloqueado_por:
        motivo = "sem sinal acima do limiar"
    elif "shadow_lock" in bloqueado_por:
        motivo = "shadow lock ativo"
    elif cooldown_ativo:
        motivo = f"cooldown ativo ate {_fmt_dt(st.cooldown_release_time)}"
    else:
        motivo = "NONE"

    return {
        "cooldown_ativo": int(cooldown_ativo),
        "cooldown_restante_barras": int(cooldown_restante_barras),
        "cooldown_restante_min": int(cooldown_restante_min),
        "bloqueado_por": bloqueado_por,
        "entrada_bloqueada_cooldown": int(entrada_bloqueada_cooldown),
        "ultimo_resultado_operacao": st.last_result if st.last_result else "NONE",
        "ultimo_resultado_pts": float(st.last_result_pts or 0),
        "loss_streak": int(st.loss_streak or 0),
        "hora_ultima_saida": _fmt_dt(st.last_exit_time),
        "hora_liberacao_cooldown": _fmt_dt(st.cooldown_release_time),
        "motivo_bloqueio_detalhado": motivo,
    }
