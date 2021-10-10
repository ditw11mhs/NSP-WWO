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

        self.nurse_first_schedule = self.generate_initial_first_schedule()
        self.nurse_second_schedule = self.nurse_first_schedule

    def generate_initial_first_schedule(self) -> np.ndarray:
        """Fungsi untuk menginisialisasi array awal jadwal 1
        yang akan dioptimalisasi menggunakan WWO

        Hard Constraint
        1. jumlah minimal perawat dalam shift harus terpenuhi
        2. tiap perawat hanya mendapatkan 1 kali shift tiap hari
        3. Jika seorang perawat bertugas pada shift malam maka
        perawat tersebut tidak bekerja pada shift pagi pada hari
        berikutnya
        4 Jika seorang perawat bekerja pada shift malam selama 2
        hari berturut-turut dalam satu minggu maka perawat
        tersebut mendapatkan libur pada hari berikutnya.

        Returns:
            nurse_array : Jadwal awal perawat (kode Int 0 1 2 3) (Jumlah Perawat x Hari)
        """
        # Memastikan sesuai Hard Constraint 2
        nurse_array_HC2 = np.random.randint(4, size=(self.unit_total_nurse * self.day))

        # Memastikan sesuai Hard Constraint 134
        nurse_array_col = np.reshape(nurse_array_HC2, (-1, self.day))

        # Memastikan sesuai Hard COnstraint 134
        nurse_array_HC134 = np.count_nonzero(nurse_array_col == 0, axis=0)
        nurse_array_HC134_Noon = np.count_nonzero(nurse_array_col == 1, axis=0)
        nurse_array_HC134_Night = np.count_nonzero(nurse_array_col == 2, axis=0)
        nurse_array_HC134_Holiday = np.count_nonzero(nurse_array_col == 3, axis=0)
        nurse_total_shift = np.vstack(
            (
                nurse_array_HC134,
                nurse_array_HC134_Noon,
                nurse_array_HC134_Night,
                nurse_array_HC134_Holiday,
            )
        ).T
        # Mencari pebedaan antara jumlah shift per hari dengan minimum shift
        array_difference = nurse_total_shift - self.unit_minimum_shift
        check_if_not_minimal = np.sum(
            np.where(array_difference[:, :3] < 0, array_difference[:, :3], 0)
        )

        # Memastikan sesuai dengan HC sembari memastikan sesuai dengan HC 3 dan 4
        while check_if_not_minimal != 0:

            # Mencari shift dengan perbedaan negatif dan positif
            array_check_n = -np.where(array_difference < 0, array_difference, 0)
            array_check_p = np.where(array_difference > 0, array_difference, 0)
            # Mencari posisi shift dengan perbedaan negatif dan positif
            array_where_n = np.nonzero(array_check_n)
            array_where_p = np.nonzero(array_check_p)
            # Membuat array yang terdiri dari kolom hari, shift, dan jumlah perbedaan)
            array_code_n = np.hstack(
                (
                    np.transpose(array_where_n),
                    array_check_n[array_where_n].reshape(-1, 1),
                )
            )
            array_code_p = np.hstack(
                (
                    np.transpose(array_where_p),
                    array_check_p[array_where_p].reshape(-1, 1),
                )
            )
            # print(nurse_array_col)

            for code in range(array_code_n.shape[0]):
                day_code = int(array_code_n[code, 0])
                shift = array_code_n[code, 1]
                total_negative = array_code_n[code, 2]

                # Mencari shift berlebih pada hari yang sama
                col_same_day = np.argwhere((array_code_p == day_code)[:, 0])
                where_day = np.squeeze(array_code_p[col_same_day]).reshape(-1, 3)

                # Untuk tiap shift berlebih
                for row_where_day in range(where_day.shape[0]):
                    if total_negative == 0:
                        break
                    # Mencari posisi perawat dengan shift berlebih di jadwal awal

                    where_in_nurse_col = np.argwhere(
                        nurse_array_col[:, day_code] == where_day[row_where_day, 1]
                    ).flatten()

                    # Untuk tiap perawat yang berlebih
                    for row_nurse_col in where_in_nurse_col:
                        if total_negative == 0:
                            break

                        # Jika merubah menjadi pagi
                        if shift == 0:
                            # Jika bukan hari awal
                            if day_code != 0:
                                # Jika sebelumnya ada shift malam maka skip
                                if nurse_array_col[row_nurse_col, day_code - 1] == 2:
                                    nurse_array_col[row_nurse_col, day_code - 1] = 1
                                    # continue
                                # Jika tidak ada shift malam maka ditukar
                                else:
                                    nurse_array_col[row_nurse_col, day_code] = shift
                                    total_negative -= 1
                            # Jika hari awal
                            else:
                                nurse_array_col[row_nurse_col, day_code] = shift
                                total_negative -= 1

                        # Jika merubah menjadi malam
                        if shift == 2:
                            # Jika sudah di hari maks, tidak perlu dicek
                            if day_code == nurse_array_col.shape[1] - 1:
                                nurse_array_col[row_nurse_col, day_code] = shift
                                total_negative -= 1

                            # Jika besoknya ada shift pagi maka skip
                            elif nurse_array_col[row_nurse_col, day_code + 1] == 0:
                                nurse_array_col[row_nurse_col, day_code + 1] = 1
                                # continue
                            # Jika besoknya tidak ada shift pagi maka ditukar
                            else:
                                nurse_array_col[row_nurse_col, day_code] = shift
                                total_negative -= 1

                        # Jika merubah menjadi sore
                        if shift == 1:
                            nurse_array_col[row_nurse_col, day_code] = shift
                            total_negative -= 1

            where_night = np.argwhere(nurse_array_col == 2)

            for night_pos in where_night:
                # HC 3
                # Cek jika tidak di hari terakhir
                if night_pos[1] + 1 < nurse_array_col.shape[1]:
                    if nurse_array_col[night_pos[0], night_pos[1] + 1] == 0:
                        # Jika besoknya pagi, maka antara diganti menjadi antara sore,malam,dan libur atau random
                        rng = np.random.randint(low=0, high=1, size=1)[0]
                        if rng == 0:
                            nurse_array_col[
                                night_pos[0], night_pos[1] + 1
                            ] = np.random.randint(low=1, high=4, size=1)[0]
                        else:
                            nurse_array_col[
                                night_pos[0], night_pos[1]
                            ] = np.random.randint(low=0, high=4, size=1)[0]
                # HC 4
                # if night_pos[1] + 2 < nurse_array_col.shape[1]:
                #     if nurse_array_col[night_pos[0], night_pos[1] + 1] == 2:
                #         rng = np.random.randint(low=0, high=1, size=1)[0]
                #         if rng == 0:
                #             nurse_array_col[night_pos[0], night_pos[1] + 2] = 3
                #         else:
                #             nurse_array_col[night_pos[0], night_pos[1]] = np.random.randint(
                #                 low=0, high=4, size=1
                #             )[0]
                #             nurse_array_col[night_pos[0], night_pos[1] + 1] = np.random.randint(
                #                 low=0, high=4, size=1
                #             )[0]

            # Memastikan sesuai Hard COnstraint 134
            nurse_array_HC134 = np.count_nonzero(nurse_array_col == 0, axis=0)
            nurse_array_HC134_Noon = np.count_nonzero(nurse_array_col == 1, axis=0)
            nurse_array_HC134_Night = np.count_nonzero(nurse_array_col == 2, axis=0)
            nurse_array_HC134_Holiday = np.count_nonzero(nurse_array_col == 3, axis=0)
            nurse_total_shift = np.vstack(
                (
                    nurse_array_HC134,
                    nurse_array_HC134_Noon,
                    nurse_array_HC134_Night,
                    nurse_array_HC134_Holiday,
                )
            ).T
            # Mencari pebedaan antara jumlah shift per hari dengan minimum shift
            array_difference = nurse_total_shift - self.unit_minimum_shift
            check_if_not_minimal2 = np.sum(
                np.where(array_difference[:, :3] < 0, array_difference[:, :3], 0)
            )
            if check_if_not_minimal == check_if_not_minimal2:
                # Memastikan sesuai Hard Constraint 2
                nurse_array_HC2 = np.random.randint(
                    4, size=(self.unit_total_nurse * self.day)
                )

                # Memastikan sesuai Hard Constraint 134
                nurse_array_col = np.reshape(nurse_array_HC2, (-1, self.day))

                # Memastikan sesuai Hard COnstraint 134
                nurse_array_HC134 = np.count_nonzero(nurse_array_col == 0, axis=0)
                nurse_array_HC134_Noon = np.count_nonzero(nurse_array_col == 1, axis=0)
                nurse_array_HC134_Night = np.count_nonzero(nurse_array_col == 2, axis=0)
                nurse_array_HC134_Holiday = np.count_nonzero(
                    nurse_array_col == 3, axis=0
                )
                nurse_total_shift = np.vstack(
                    (
                        nurse_array_HC134,
                        nurse_array_HC134_Noon,
                        nurse_array_HC134_Night,
                        nurse_array_HC134_Holiday,
                    )
                ).T
                # Mencari pebedaan antara jumlah shift per hari dengan minimum shift
                array_difference = nurse_total_shift - self.unit_minimum_shift
                check_if_not_minimal = np.sum(
                    np.where(array_difference[:, :3] < 0, array_difference[:, :3], 0)
                )
            else:
                check_if_not_minimal = check_if_not_minimal2
        self.nurse_array_col = nurse_array_col
        self.nurse_array_col = np.where(self.nurse_array_col.astype(str)=="0","Pagi",self.nurse_array_col.astype(str))
        self.nurse_array_col = np.where(self.nurse_array_col.astype(str)=="1","Sore",self.nurse_array_col.astype(str))
        self.nurse_array_col = np.where(self.nurse_array_col.astype(str)=="2","Malam",self.nurse_array_col.astype(str))
        self.nurse_array_col = np.where(self.nurse_array_col.astype(str)=="3","Libur",self.nurse_array_col.astype(str))
        # nurse_array = (
        #     (nurse_array_col.flatten()[:, None] == np.arange(4)) * 1
        # ).flatten()
        
        nurse_array = nurse_array_col.flatten()
        return nurse_array

    def cost(self, nurse_array) -> float:
        """Fungsi untuk menghitung total cost dari model NSP

        Returns:
            cost: Nilai cost dari model NSP
        """
        # nurse_array = nurse_array.reshape(self.unit_total_nurse, 4 * self.day)
        # nurse_array = np.round(nurse_array)
        
        nurse_array = nurse_array.reshape(-1,self.day)

        cost_minimum_shift = self.hard_constraint_cost_minimum_shift(nurse_array)
        cost_one_per_day = self.hard_constraint_cost_one_per_day(nurse_array)
        cost_night_day = self.hard_constraint_cost_night_day(nurse_array)

        cost_noon_shift = self.soft_constraint_cost_noon_shift(nurse_array)
        cost_morning_shift = self.soft_constraint_cost_morning_shift(nurse_array)
        cost_night_holiday_noon = self.soft_constraint_cost_night_holiday_noon(
            nurse_array
        )
        cost = (
            self.hard_constraint_multiplier
            * (cost_minimum_shift + cost_one_per_day + cost_night_day)
            + cost_noon_shift
            + cost_morning_shift
            + cost_night_holiday_noon
        )

        return cost

    def hard_constraint_cost_minimum_shift(self, nurse_array) -> int:
        """Fungsi untuk menghitung total cost dari hard constraint "minimum shift terpenuhi"

        Returns:
            cost_minimum_shift: Total cost "minimum shift"
        """
        # # Menghitung total perawat per shift per hari (sum ke bawah)
        # array_shift_sum = np.sum(nurse_array, axis=0)
        # # Membuat array total perawat per shift per hari menjadi kolom (Hari x 4)
        # array_rearange = np.reshape(array_shift_sum, (-1, 4))
        # # Menggunakan 3 kolom awal array sebelumnya (karena kolom 4 libur sehingga tidak dihitung)
        # # dan memastikan apakah nilainya sesuai dengan shift minimum
        # array_difference = array_rearange - self.unit_minimum_shift
        # array_check = -np.where(array_difference < 0, array_difference, 0)
        # # Menghitung total cost
        # cost_minimum_shift = np.sum(array_check)
        
        nurse_array_HC134 = np.count_nonzero(nurse_array == 0, axis=0)
        nurse_array_HC134_Noon = np.count_nonzero(nurse_array == 1, axis=0)
        nurse_array_HC134_Night = np.count_nonzero(nurse_array == 2, axis=0)
        nurse_array_HC134_Holiday = np.count_nonzero(nurse_array == 3, axis=0)
        nurse_total_shift = np.vstack(
            (
                nurse_array_HC134,
                nurse_array_HC134_Noon,
                nurse_array_HC134_Night,
                nurse_array_HC134_Holiday,
            )
        ).T
        array_difference = nurse_total_shift - self.unit_minimum_shift
        check_if_not_minimal = np.sum(
            np.where(array_difference[:, :3] < 0, array_difference[:, :3], 0)
        )
        cost_minimum_shift = -check_if_not_minimal
        return cost_minimum_shift

    def hard_constraint_cost_one_per_day(self, nurse_array) -> int:
        """Fungsi untuk menghitung total cost dari hard constraint "satu shift per hari"

        Returns:
            cost_one_per_day: Total cost "satu shift per hari"
        """
        # # Membuat array menjadi kolom (Total Perawat*Hari x 4)
        # array_compute = np.reshape(nurse_array, (-1, 4))
        # # Menghitung total (ke samping) untuk mendapatkan jumlah shift per hari
        # array_compute_sum = np.sum(array_compute, axis=1)
        # # Memastikan apakah jumlah shift per hari lebih dari 1
        # array_check = (array_compute_sum > 1) * 1
        # # Menghitung total cost
        # cost_one_per_day = np.sum(array_check)
        
        # Karena sudah pasti satu shift per harinya 
        cost_one_per_day = 0 
        return cost_one_per_day

    def hard_constraint_cost_night_day(self, nurse_array):
        """FUngsi untuk menghitung total cost dari hard constraint "Jika seorang perawat bertugas pada shift malam maka
            perawat tersebut tidak bekerja pada shift pagi pada hari
            berikutnya"

        Returns:
            cost_night_day : cost perawat dengan shift malam diikuti dengan shift pagi
        """
        # array_night = nurse_array[:, 2:-2:4] * nurse_array[:, 4::4]
        # cost_night_day = np.sum(array_night)
        
        where_night = np.argwhere(nurse_array==2)
        where_night_plus = where_night+[0,1]
        where_more_than_max = np.argwhere(where_night_plus[:,1]>self.day-1)
        where_night_plus = np.delete(where_night_plus,(where_more_than_max*2,where_more_than_max*2+1)).reshape(-1,2,)
        nurse_array_plus = nurse_array[where_night_plus[:,0],where_night_plus[:,1]]
        cost_night_day = np.sum(np.where(nurse_array_plus==0,1,0))
        return cost_night_day

    def soft_constraint_cost_noon_shift(self, nurse_array) -> int:
        """Fungsi untuk menghitung total cost dari soft constraint "menghindari setelah shift siang diikuti dengan shift pagi
            dihari berikutnya (dikarenakan perawat akan kelelahan
            secara fisik karena terus bekerja pada shift siang yaitu
            dimulai pukul 14.00 hingga 21.00 lalu dilanjutkan pada
            shift pagi dimulai pukul 07.00 pagi di hari berikutnya.)
            "

        Returns:
            cost_noon_shift: cost soft constraint shift sore
        """
        # Membuat array sore (hari 1 -> Total Hari-1)
        # array_noon = nurse_array[:, 1:-3:4] * nurse_array[:, 4::4]
        # cost_noon_shift = np.sum(array_noon)
        
        where_noon = np.argwhere(nurse_array==1)
        where_noon_plus = where_noon+[0,1]
        where_more_than_max = np.argwhere(where_noon_plus[:,1]>self.day-1)
        where_noon_plus = np.delete(where_noon_plus,(where_more_than_max*2,where_more_than_max*2+1)).reshape(-1,2,)
        nurse_array_plus = nurse_array[where_noon_plus[:,0],where_noon_plus[:,1]]
        cost_noon_shift = np.sum(np.where(nurse_array_plus==0,1,0))
        return cost_noon_shift

    def soft_constraint_cost_morning_shift(self, nurse_array) -> int:
        """Fungsi untuk menghitung total cost dari soft constraint "Menghindari setelah shift pagi diikuti dengan shift malam
            dihari berikutnya (agar perawat tidak mendapatkan pola
            bekerja seperti ini dikarenakan waktu istirahat yang
            didapatkan terlalu panjang sehingga akan menurunkan
            motivasi dari perawat untuk bekerja.)


        Returns:
            cost_morning_shift: cost soft constraint shift pagi
        """
        # array_morning = nurse_array[:, :-4:4] * nurse_array[:, 6::4]
        # cost_morning_shift = np.sum(array_morning)
        
        where_morning = np.argwhere(nurse_array==0)
        where_morning_plus = where_morning+[0,1]
        where_more_than_max = np.argwhere(where_morning_plus[:,1]>self.day-1)
        where_morning_plus = np.delete(where_morning_plus,(where_more_than_max*2,where_more_than_max*2+1)).reshape(-1,2,)
        nurse_array_plus = nurse_array[where_morning_plus[:,0],where_morning_plus[:,1]]
        cost_morning_shift = np.sum(np.where(nurse_array_plus==2,1,0))
        return cost_morning_shift

    def soft_constraint_cost_night_holiday_noon(self, nurse_array) -> int:
        """Fungsi untuk menghitung total cost dari soft constraint "memberikan jadwal jaga sore setelah hari libur yang
            didapat setelah jaga malam

        Returns:
            cost_night_holiday_noon: cost soft constraint malam libur sore
        """
        # array_night_holiday_noon = (
        #     nurse_array[:, 2:-8:4] * nurse_array[:, 7:-4:4] * nurse_array[:, 9::4]
        # )
        # cost_night_holiday_noon = -np.sum(array_night_holiday_noon)
        
        where_night = np.argwhere(nurse_array==2)
        where_night_plus = where_night+[0,1]
        where_night_plus_plus = where_night_plus+[0,1]
        where_more_than_max = np.argwhere(where_night_plus_plus[:,1]>self.day-1)
        where_night_plus = np.delete(where_night_plus,(where_more_than_max*2,where_more_than_max*2+1)).reshape(-1,2,)
        where_night_plus_plus = np.delete(where_night_plus_plus,(where_more_than_max*2,where_more_than_max*2+1)).reshape(-1,2,)
        nurse_array_plus = nurse_array[where_night_plus[:,0],where_night_plus[:,1]]
        nurse_array_plus_plus = nurse_array[where_night_plus_plus[:,0],where_night_plus_plus[:,1]]
        cost_night_holiday_noon=-np.sum(np.where(nurse_array_plus==3,1,0)*np.where(nurse_array_plus_plus==1,1,0))
        return cost_night_holiday_noon


class WWO:
    def __init__(
        self,
        NSP: NSP_Class,
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
        x_population=1,
    ) -> None:
        """Inisialisasi WWO dengan NSP

        Args:
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
        # print(wave_population_list)
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
            cost_list.append(wave.cost(wave.nurse_second_schedule))
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
            wave_population_list[min_index].nurse_second_schedule,
            wave_population_cost_list[min_index],
        )
        # Inisialisasi nilai beta (untuk nanti diupdate setiap iterasi secara linear)
        beta = self.beta_max
        self.best_fit_iteration = []
        self.best_fit_iteration.append(wave_population_cost_list[min_index])
        # Iterasi berdasarkan jumlah iterasi maksimal
        for iteration in range(self.iteration):
            # new_fit_counter = 0
            # best_fit_counter = 0
            # not_found_counter = 0
            # if best_fit == 0.0:
            #     break
            # Iterasi untuk tiap gelombang dalam populasi
            for index, wave in enumerate(wave_population_list):
                new_pos, new_fit = self.propagation(wave)
                # print(new_pos,new_fit)
                if new_fit < wave_population_cost_list[index]:
                    # new_fit_counter += 1
                    wave.nurse_second_schedule, wave_population_cost_list[index] = (
                        new_pos,
                        new_fit,
                    )
                    wave_height[index] = self.hmax
                    if new_fit < best_fit and index != min_index:
                        # best_fit_counter += 1
                        new_pos, new_fit, wave_length[index] = self.breaking(
                            new_pos, new_fit, wave_length[index], beta, wave
                        )
                        best_pos, best_fit = new_pos, new_fit
                        
                        # print(best_fit)
                else:
                    # not_found_counter += 1
                    wave_height[index] -= 1
                    if wave_height[index] == 0:
                        fit_old = wave_population_cost_list[index]
                        (
                            wave.nurse_second_schedule,
                            wave_population_cost_list[index],
                        ) = self.refraction(wave.nurse_second_schedule, best_pos, wave)
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
            # print(
            #     f"""
            #       new fit = {new_fit_counter}
            #       best fit = {best_fit_counter}
            #       not found = {not_found_counter}
            #       """
            # )
            self.best_fit_iteration.append(best_fit)
        # best_pos = best_pos.reshape(-1, 4)
        # where_one_col = np.argwhere(best_pos == 1)[:, 1]
        # best_pos = where_one_col.reshape(self.NSP.unit_total_nurse, self.NSP.day)
        best_pos = best_pos.reshape(self.NSP.unit_total_nurse,self.NSP.day)
        best_pos = best_pos.astype(int)
        best_pos = np.where(best_pos.astype(str)=="0","Pagi",best_pos.astype(str))
        best_pos = np.where(best_pos.astype(str)=="1","Sore",best_pos.astype(str))
        best_pos = np.where(best_pos.astype(str)=="2","Malam",best_pos.astype(str))
        best_pos = np.where(best_pos.astype(str)=="3","Libur",best_pos.astype(str))
        return best_pos, best_fit

    def propagation(self, wave: NSP_Class) -> tuple:
        # print("propagate")
        l = np.abs(self.upper_bound - self.lower_bound)
        new_pos = (
            wave.nurse_second_schedule
            + np.random.uniform(-1, 1, size=wave.nurse_second_schedule.shape)
            * l
            * self.lambd
        )
        new_pos = self.boundary_handle(new_pos)
        new_fit = wave.cost(new_pos)
        return new_pos, new_fit

    def boundary_handle(self, new_pos) -> np.ndarray:
        """Fungsi untuk menghandle nilai yang melewati batas

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
