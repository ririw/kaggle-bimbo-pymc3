import luigi
import luigi.file
from sklearn import *
import sqlite3
import pandas

class NMFDecomposition(luigi.Task):
    factors = luigi.Parameter()

    def output(self):
        factor_name = '-'.join(self.factors.split(','))
        return luigi.file.LocalTarget(path='/tmp/decompositions/nmf_{}.msgpack'.format(factor_name))

    def run(self):
        con = sqlite3.connect("/tmp/data.sqlite3")
        query = """
            SELECT {},
                   week_num,
                   avg(adjusted_demand) as adjusted_demand
              FROM data
             WHERE adjusted_demand is not NULL
                   and week_num < 8
          GROUP BY {},week_num
        """.format(self.factors, self.factors)
        data = pandas.read_sql(query, con=con)
        week_progression = data.group_by(self.factors.split(',') + ['week_num']).adjusted_demand.mean().unstack()
        nmf = decomposition.NMF(n_components=3)
        results = nmf.fit_transform(week_progression)
        col_names = ['nmf_{}_{}'.format(i, '-'.join(self.factors.split(','))) for i in range(nmf.n_components)]
        result_frame = pandas.DataFrame(results, columns=col_names, index=week_progression.index)
        with open(self.output().path, 'wb') as f:
            result_frame.to_msgpack(f, compress='zlib')

