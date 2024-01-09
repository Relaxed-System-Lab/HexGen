class CostModel:
    def __init__(self, alpha, beta, H, B_type, s_in_i, s_out_i, b_i, m_d):
        self.alpha = alpha
        self.beta = beta
        self.H = H
        self.B_type = B_type
        self.s_in_i = s_in_i
        self.s_out_i = s_out_i
        self.b_i = b_i
        self.m_d = m_d

    def compute_prompting_phase_no_tensor_parallelism(self, l_j, bsz):
        self.b_i = bsz
        return (12 * self.H**2 * self.B_type / self.m_d + 24 * self.b_i * self.s_in_i * self.H**2 / self.m_d) * l_j

    def compute_token_generation_phase_no_tensor_parallelism(self, l_j, bsz):
        self.b_i = bsz
        return (12 * self.H**2 * self.B_type / self.m_d + 24 * self.b_i * self.H**2 / self.m_d) * l_j

    def compute_prompting_phase_with_tensor_parallelism(self, l_j, D_TP_j):
        return (12 * self.H**2 * self.B_type / (D_TP_j * self.m_d)) + (24 * self.b_i * self.s_in_i * self.H**2 / (D_TP_j * self.m_d)) * l_j

    def compute_token_generation_phase_with_tensor_parallelism(self, l_j, D_TP_j):
        return (12 * self.H**2 * self.B_type / (D_TP_j * self.m_d)) + (24 * self.b_i * self.H**2 / (D_TP_j * self.m_d)) * l_j

    def compute_prompting_phase_communication_with_tensor_parallelism(self, l_j, D_TP_j, bsz):
        self.b_i = bsz
        max_val = self.alpha + self.b_i * self.s_in_i * self.H * self.B_type / (D_TP_j * self.beta)
        return 2 * (D_TP_j - 1) * max_val * 2 * l_j

    def compute_token_generation_phase_communication_with_tensor_parallelism(self, l_j, D_TP_j, bsz):
        self.b_i = bsz
        max_val = self.alpha + self.b_i * self.H * self.B_type / (D_TP_j * self.beta)
        return 2 * (D_TP_j - 1) * max_val * 2 * l_j

    def compute_pipeline_parallelism_prompting_phase(self, D_j1, bsz):
        self.b_i = bsz
        if D_j1 == 1:
            return self.alpha + self.b_i * self.s_in_i * self.H * self.B_type / self.beta
        else:
            min_val = self.alpha + self.b_i * self.s_in_i * self.H * self.B_type / self.beta + (D_j1 - 1) * (self.alpha + self.b_i * self.s_in_i * self.H * self.B_type / (D_j1 * self.beta))
            max_val = (D_j1 - 1) * (self.alpha + self.b_i * self.s_in_i * self.H * self.B_type / (D_j1 * self.beta))
            return min_val + max_val

    def compute_pipeline_parallelism_token_generation_phase(self, D_j1, bsz):
        self.b_i = bsz
        if D_j1 == 1:
            return self.alpha + self.b_i * self.H * self.B_type / self.beta
        else:
            min_val = self.alpha + self.b_i * self.H * self.B_type / self.beta + (D_j1 - 1) * (self.alpha + self.b_i * self.H * self.B_type / (D_j1 * self.beta))
            max_val = (D_j1 - 1) * (self.alpha + self.b_i * self.H * self.B_type / (D_j1 * self.beta))
            return min_val + max_val
    
    def compute_memory_usage(self, l_j, D_j, bsz):
        self.b_i = bsz
        memory_usage = ((12 * self.H**2 * self.B_type / D_j) + 2 * self.b_i * (self.s_in_i + self.s_out_i) * self.H * self.B_type) * l_j + 4 * self.b_i * (self.s_in_i + self.s_out_i) * self.H * self.B_type
        return (memory_usage) / (1024 ** 3)