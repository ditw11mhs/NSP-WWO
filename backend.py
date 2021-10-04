import numpy as np


class NSP_Class:
    def __init__(
        self,
        day: int,
        units_name: str,
        unit_total_nurse: int,
        unit_minimum_shift: np.ndarray,
        hard_constraint_multiplier: int,
    ) -> None:
        """Class Nurse Scheduling Problem yang berfungsi sebagai kontainer penyimpanan
        data-data yang dibutuhkan untuk menjalankan algoritma WWO. Class ini juga berfungsi
        untuk mencari nilai cost berdasarkan hard dan soft constraint.

        Args:
            day (int): Jumlah hari yang akan dijadwalkan
            units_name (str): Nama Unit
            unit_total_nurse (int): Jumlah Perawat yang ada di unit
            unit_minimum_shift (np.ndarray): Jumlah minimum perawat tiap shift (1 x 3)
            hard_constraint_multiplier (int): Koefisien pengali hard constraint
        """
        self.day = day
        self.units_name = units_name
        self.unit_total_nurse = unit_total_nurse
        self.unit_minimum_shift = unit_minimum_shift
        self.hard_constraint_multiplier = hard_constraint_multiplier

        self.nurse_array = self.generate_initial_nurse_array()

    def generate_initial_nurse_array(self) -> np.ndarray:
        """Fungsi untuk menginisialisasi array awal jadwal

        Returns:
            nurse_array : Jadwal awal perawat (Jumlah Perawat x 4*Hari)
        """
        nurse_array = np.random.randint(2, size=(self.unit_total_nurse * 4 * self.day))
        return nurse_array

    def cost(self, nurse_array) -> float:
        """Fungsi untuk menghitung total cost dari model NSP

        Returns:
            cost: Nilai cost dari model NSP
        """
        nurse_array = nurse_array.reshape(self.unit_total_nurse, 4 * self.day)
        nurse_array = np.round(nurse_array)
        cost_minimum_shift = self.hard_constraint_cost_minimum_shift(nurse_array)
        cost_one_per_day = self.hard_constraint_cost_one_per_day(nurse_array)
        cost_noon_shift = self.soft_constraint_cost_noon_shift(nurse_array)
        cost_morning_shift = self.soft_constraint_cost_morning_shift(nurse_array)
        cost = (
            self.hard_constraint_multiplier * (cost_minimum_shift + cost_one_per_day)
            + cost_noon_shift
            + cost_morning_shift
        )

        return cost

    def hard_constraint_cost_minimum_shift(self, nurse_array) -> int:
        """Fungsi untuk menghitung total cost dari hard constraint "minimum shift terpenuhi"

        Returns:
            cost_minimum_shift: Total cost "minimum shift"
        """
        # Menghitung total perawat per shift per hari (sum ke bawah)
        array_shift_sum = np.sum(nurse_array, axis=0)
        # Membuat array total perawat per shift per hari menjadi kolom (Hari x 4)
        array_rearange = np.reshape(array_shift_sum, (-1, 4))
        # Menggunakan 3 kolom awal array sebelumnya (karena kolom 4 libur sehingga tidak dihitung)
        # dan memastikan apakah nilainya sesuai dengan shift minimum
        array_difference = array_rearange[:, :3] - self.unit_minimum_shift
        array_check = -np.where(array_difference < 0, array_difference, 0)
        # Menghitung total cost
        cost_minimum_shift = np.sum(array_check)
        return cost_minimum_shift

    def hard_constraint_cost_one_per_day(self, nurse_array) -> int:
        """Fungsi untuk menghitung total cost dari hard constraint "satu shift per hari"

        Returns:
            cost_one_per_day: Total cost "satu shift per hari"
        """
        # Membuat array menjadi kolom (Total Perawat*Hari x 4)
        array_compute = np.reshape(nurse_array, (-1, 4))
        # Menghitung total (ke samping) untuk mendapatkan jumlah shift per hari
        array_compute_sum = np.sum(array_compute, axis=1)
        # Memastikan apakah jumlah shift per hari lebih dari 1
        array_check = (array_compute_sum > 1) * 1
        # Menghitung total cost
        cost_one_per_day = np.sum(array_check)
        return cost_one_per_day

    def soft_constraint_cost_noon_shift(self, nurse_array) -> int:
        """Fungsi untuk menghitung total cost dari soft constraint "Menghindari setelah shift sore diikuti dengan shift pagi atau shift
        malam di hari berikutnya."


        Returns:
            cost_noon_shift: Dictionary berisi total cost soft constraint shift sore per unit
        """
        # Membuat array sore (hari 1 -> Total Hari-1)
        array_noon = nurse_array[:, 1:-4:4]
        # Membuat array pagi (hari 2 -> Total Hari)
        array_morning = nurse_array[:, 4::4]
        # Membuat array malam (hari 2 -> Total Hari)
        array_night = nurse_array[:, 6::4]
        # Membuat array untuk memastikan apakah shift sore
        # diikuti oleh shift pagi di hari berikutnya
        array_check_morning = np.logical_and(
            array_noon, array_morning
        )  # Menggunakan logika AND
        # Membuat array untuk memastikan apakah shift sore
        # diikuti oleh shift malam di hari berikutnya
        array_check_night = np.logical_and(
            array_noon, array_night
        )  # Menggunakan logika AND
        # Menghitung total cost
        cost_noon_shift = np.sum(array_check_morning + array_check_night)
        return cost_noon_shift

    def soft_constraint_cost_morning_shift(self, nurse_array) -> int:
        """Fungsi untuk menghitung total cost dari soft constraint "Menghindari setelah shift pagi diikuti dengan shift sore atau shift
        malam dihari berikutnya


        Returns:
            cost_morning_shift: Dictionary berisi total cost soft constraint shift pagi per unit
        """
        # Membuat array pagi (hari 1 -> Total Hari-1)
        array_morning = nurse_array[:, :-4:4]
        # Membuat array sore (hari 2 -> Total Hari)
        array_noon = nurse_array[:, 5::4]
        # Membuat array malam (hari 2 -> Total Hari)
        array_night = nurse_array[:, 6::4]
        # Membuat array untuk memastikan apakah shift sore
        # diikuti oleh shift pagi di hari berikutnya
        array_check_noon = np.logical_and(
            array_morning, array_noon
        )  # Menggunakan logika AND
        # Membuat array untuk memastikan apakah shift sore
        # diikuti oleh shift malam di hari berikutnya
        array_check_night = np.logical_and(
            array_morning, array_night
        )  # Menggunakan logika AND
        # Menghitung total cost
        cost_morning_shift = np.sum(array_check_noon + array_check_night)
        return cost_morning_shift


class WWO:
    def __init__(
        self,
        NSP: NSP_Class,
        x_population: int,
        iteration: int,
        hmax,
        lambd,
        alpha,
        epsilon,
        beta_max,
        beta_min,
        k_max,
        upper_bound: float,
        lower_bound: float,
    ) -> None:
        """Inisialisasi WWO

        Args:
            NSP: NSP_Class
            hmax: int
            lambd: int
            alpha: float
            epsilon: float
            beta_max: float
            beta_min: float
            k_max: int
            upper_bound: float
            lower_bound: float
        """
        self.NSP = NSP
        self.x_population = x_population
        self.iteration = iteration
        self.hmax = hmax
        self.lambd = lambd
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.k_max = k_max
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def initialize_population(self) -> list:
        """Fungsi untuk menginisialisasi populasi awal

        Returns:
            wave_population_list: List yang berisikan class untuk tiap populasinya
        """
        wave_population_list = []
        for x in range(self.x_population):
            wave_population_list.append(self.NSP)
        return wave_population_list

    def cost_function(self, wave_population_list: list) -> list:
        """Fungsi untuk menghitung cost dari setiap wave

        Args:
            wave_population_list: List berisi wave

        Returns:
            cost_list: List berisi total cost dari setiap wave
        """
        cost_list = []
        for wave in wave_population_list:
            cost_list.append(wave.cost(wave.nurse_array))
        return cost_list

    def optimize(self) -> tuple:

        # Inisialisasi populasi awal
        wave_population_list = self.initialize_population()
        # Inisialisasi cost awal
        wave_population_cost_list = self.cost_function(wave_population_list)
        # Inisialisasi panjang gelombang awal
        wave_length = np.full(self.x_population, self.lambd)
        # Inisialisasi tinggi gelombang awal
        wave_height = np.full(self.x_population, self.hmax)
        # Indexing untuk mencari cost terkecil
        min_index = np.argmin(wave_population_cost_list)
        best_pos, best_fit = (
            wave_population_list[min_index].nurse_array,
            wave_population_cost_list[min_index],
        )
        # Inisialisasi nilai beta (untuk nanti diupdate setiap iterasi secara linear)
        beta = self.beta_max

        # Iterasi berdasarkan jumlah iterasi maksimal
        for iteration in range(self.iteration):
            new_fit_counter = 0
            best_fit_counter = 0
            not_found_counter = 0
            if best_fit == 0.0:
                break
            # Iterasi untuk tiap gelombang dalam populasi
            for index, wave in enumerate(wave_population_list):
                new_pos, new_fit = self.propagation(wave)
                # print(new_pos,new_fit)
                if new_fit < wave_population_cost_list[index]:
                    new_fit_counter += 1
                    wave.nurse_array, wave_population_cost_list[index] = (
                        new_pos,
                        new_fit,
                    )
                    wave_height[index] = self.hmax
                    if new_fit < best_fit and index != min_index:
                        best_fit_counter += 1
                        new_pos, new_fit, wave_length[index] = self.breaking(
                            new_pos, new_fit, wave_length[index], beta, wave
                        )
                        best_pos, best_fit = new_pos, new_fit
                        print(best_fit)
                else:
                    not_found_counter += 1
                    wave_height[index] -= 1
                    if wave_height[index] == 0:
                        fit_old = wave_population_cost_list[index]
                        (
                            wave.nurse_array,
                            wave_population_cost_list[index],
                        ) = self.refraction(wave.nurse_array, best_pos, wave)
                        wave_height[index] = self.hmax
                        wave_length[index] = self.set_wave_length(
                            wave_length[index],
                            fit_old,
                            wave_population_cost_list[index],
                        )

            min_index, max_index = np.argmin(wave_population_cost_list), np.argmax(
                wave_population_cost_list
            )
            wave_length = self.update_wave_length(
                wave_length,
                wave_population_cost_list,
                wave_population_cost_list[max_index],
                wave_population_cost_list[min_index],
            )

            beta = self.update_beta(iteration)
            print(
                f"""
                  new fit = {new_fit_counter}
                  best fit = {best_fit_counter}
                  not found = {not_found_counter}
                  """
            )
            # if iteration%10==0:

        return best_pos, best_fit

    def propagation(self, wave: NSP_Class) -> tuple:
        # print("propagate")
        l = np.abs(self.upper_bound - self.lower_bound)
        new_pos = (
            wave.nurse_array
            + np.random.uniform(-1, 1, size=wave.nurse_array.shape) * l * self.lambd
        )
        new_pos = self.boundary_handle(new_pos)
        new_fit = wave.cost(new_pos)
        return new_pos, new_fit

    def boundary_handle(self, new_pos) -> np.ndarray:
        """Fungsi untuk menghandle nilai melewati batas

        Args:
            new_pos: np.ndarray

        Returns:
            new_pos: np.ndarray
        """
        new_pos = np.where(
            np.logical_or(new_pos > self.upper_bound, new_pos < self.lower_bound),
            np.random.uniform(self.lower_bound, self.upper_bound),
            new_pos,
        )

        return new_pos

    def breaking(self, new_pos, new_fit, wave_length, beta, wave) -> tuple:
        print("breaking")
        k = np.random.randint(1, self.k_max)
        temp = np.random.permutation(new_pos.shape[0])[:k]
        for i in range(k):
            temp_pos = new_pos.copy()
            d = temp[i]
            temp_pos[d] = new_pos[d] + np.random.normal(0, 1) * beta * np.fabs(
                self.upper_bound - self.lower_bound
            )
            self.boundary_handle(temp_pos)
            temp_fit = wave.cost(temp_pos)

            if temp_fit < new_fit:
                new_pos[d] = temp_pos[d]
                wave_length = self.set_wave_length(wave_length, new_fit, temp_fit)
                new_fit = temp_fit
        return new_pos, new_fit, wave_length

    def set_wave_length(self, wave_length, fit_old, fit) -> float:
        return wave_length * fit_old / (fit + self.epsilon)

    def refraction(self, pos, best_pos, wave):
        # print("refract")
        mu = (best_pos + pos) / 2
        sigma = np.fabs(best_pos - pos) / 2
        new_pos = np.random.normal(mu, sigma, size=pos.shape)
        new_pos = self.boundary_handle(new_pos)
        new_fit = wave.cost(new_pos)
        return new_pos, new_fit

    def update_wave_length(self, wave_length, wave_cost_list, max_cost, min_cost):
        return wave_length * np.power(
            self.alpha,
            -(wave_cost_list - min_cost + self.epsilon)
            / (max_cost - min_cost + self.epsilon),
        )

    def update_beta(self, index):
        return self.beta_max - (self.beta_max - self.beta_min) * index / self.iteration
